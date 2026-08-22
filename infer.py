from __future__ import annotations

import argparse
import inspect
import time
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from recirculation import RecirculationConfig, recirculate


EXAMPLE_QUERIES = (
    "Explain why the daytime sky is blue but sunsets often appear red. Connect the explanation to scattering, wavelength, and the distance sunlight travels through the atmosphere.",
    "A city wants to reduce downtown traffic without making commuting harder for low-income workers. Compare congestion pricing, improved public transit, and parking restrictions, then recommend a phased policy with safeguards and measurable success criteria.",
    "Maya manages a project originally due Friday. The client moves it to Wednesday, an engineer reports a two-day blocker, and a required reviewer is unavailable Tuesday. Develop a realistic recovery plan, identify assumptions, and explain what Maya should communicate to each stakeholder.",
    "A company claims that productivity increased after employees returned to the office, so remote work must reduce productivity. Critique this inference, propose plausible confounders, and design a stronger evaluation that could support a causal conclusion.",
    "Design a fair procedure for allocating five emergency shelter beds among twelve eligible people when needs differ and information is incomplete. Explain the values behind your procedure and how appeals or new evidence should be handled.",
    "Fred took his fishing pole to the bank of a river. Later, a friend texted that she would meet him at the bank to discuss a loan. Analyze the ambiguity, explain which interpretation each person may hold, and propose a message that prevents a costly misunderstanding.",
    "All roses are flowers, some flowers fade quickly, and no quickly fading plant survives a frost. Explain exactly what can and cannot be inferred about roses, then give two additional premises that would support different conclusions.",
    "A small software team must choose between shipping a fragile feature this week or delaying it for testing while a competitor is launching a similar product. Build a decision framework, evaluate the main risks, and recommend a course of action under clearly stated assumptions.",
    "Plan how to prepare tea for six guests when there is one kettle, four clean mugs, two dirty mugs, and one guest avoids caffeine. Include ordering, resource constraints, and a contingency if the kettle stops working.",
    "A store is considering a 25% discount followed by a loyalty reward, but margins are thin and customers respond differently to promotions. Explain how the store should evaluate profitability, customer behavior, and long-term effects before choosing a promotion design.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with one-step residual-stream recirculation."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Free-form prompt. Overrides --query-index when provided.",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        choices=range(1, len(EXAMPLE_QUERIES) + 1),
        default=1,
        metavar="N",
        help="Select a built-in query by its 1-based index (default: 1).",
    )
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="Print the built-in queries and exit.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    parser.add_argument("--destination", type=int, default=4)
    parser.add_argument(
        "--source",
        type=int,
        default=-1,
        help="Source block index; -1 selects the last block (default: -1).",
    )
    # alpha: weight of the source residual stream in the convex combination. 
    parser.add_argument("--alpha", type=float, default=0.5)
    # beta: weight of the destination residual stream in the convex combination. 
    # If None, defaults to 1 - alpha.
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Defaults to 1 - alpha.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the Recirculation OFF generation run.",
    )
    parser.add_argument(
        "--tempshard",
        type=int,
        default=2,
        help="Temporarily split MoE experts into this many shards; 2 is supported.",
    )
    parser.add_argument(
        "--expert-overlap",
        type=float,
        default=0.2,
        help="Fraction of experts shared by both temporal shards (default: 0.25).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Use greedy decoding at 0, sampling above 0.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Single-device fallback used when --device-map is none.",
    )
    parser.add_argument(
        "--device-map",
        choices=("balanced", "auto", "balanced_low_0", "sequential", "none"),
        default="balanced",
        help="Transformers device map. 'balanced' shards layers across all GPUs.",
    )
    parser.add_argument(
        "--gpu-memory",
        default="46GiB",
        help="Maximum model memory per GPU when sharding (default: 46GiB).",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_source(source: int, num_blocks: int) -> int:
    return num_blocks - 1 if source == -1 else source


def find_decoder_blocks(model: nn.Module) -> Sequence[nn.Module]:
    candidate_paths = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("transformer", "blocks"),
    )
    for path in candidate_paths:
        value: Any = model
        for attribute in path:
            value = getattr(value, attribute, None)
            if value is None:
                break
        if isinstance(value, (nn.ModuleList, list, tuple)) and all(
            isinstance(block, nn.Module) for block in value
        ):
            return value

    raise ValueError(
        "Could not locate the decoder blocks. Add the model's ModuleList path "
        "to find_decoder_blocks()."
    )


def resolve_shared_expert_count(num_experts: int, overlap: float) -> int:
    target = num_experts * overlap
    shared_count = round(target)
    if (num_experts - shared_count) % 2 != 0:
        shared_count += 1 if target > shared_count else -1
    return shared_count


class TemporalExpertShards:
    def __init__(
        self, blocks: Sequence[nn.Module], seed: int, overlap: float = 0.25
    ) -> None:
        if not 0.0 <= overlap < 1.0:
            raise ValueError("Expert overlap must be in the range [0, 1).")
        self._enabled = False
        self._subset = 0
        self._expert_subsets: dict[nn.Module, tuple[Tensor, Tensor]] = {}
        self._handles: list[Any] = []
        generator = torch.Generator().manual_seed(seed)

        for block in blocks:
            mlp = getattr(block, "mlp", None)
            router = getattr(mlp, "gate", None)
            experts = getattr(mlp, "experts", None)
            num_experts = getattr(router, "num_experts", None)
            top_k = getattr(router, "top_k", None)
            if not (
                isinstance(router, nn.Module)
                and isinstance(experts, nn.Module)
                and isinstance(num_experts, int)
                and isinstance(top_k, int)
            ):
                continue
            shared_count = resolve_shared_expert_count(num_experts, overlap)
            exclusive_count = num_experts - shared_count
            shard_size = shared_count + exclusive_count // 2
            if top_k > shard_size:
                raise ValueError(
                    f"The router selects {top_k} experts, but each temporal shard "
                    f"contains only {shard_size}."
                )

            permutation = torch.randperm(num_experts, generator=generator)
            shared = permutation[:shared_count]
            exclusive = permutation[shared_count:]
            split = exclusive_count // 2
            subsets = (
                torch.cat((shared, exclusive[:split])),
                torch.cat((shared, exclusive[split:])),
            )
            self._expert_subsets[router] = subsets
            self._handles.append(router.register_forward_hook(self._route_with_subset))

        print(
            f"Temporal expert sharding with {overlap:.0%} overlap for "
            f"{len(self._handles)} MoE layers."
        )

    @property
    def is_moe(self) -> bool:
        return bool(self._handles)

    def select(self, subset: int) -> None:
        if subset not in (0, 1):
            raise ValueError("Temporal expert subset must be 0 or 1.")
        self._enabled = True
        self._subset = subset

    def disable(self) -> None:
        self._enabled = False

    def _route_with_subset(
        self, router: nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> Any:
        if not self._enabled:
            return output
        if not (
            isinstance(output, tuple)
            and len(output) == 3
            and all(isinstance(value, Tensor) for value in output)
        ):
            raise TypeError(
                f"Unsupported MoE router output from {type(router).__name__}; "
                "expected (logits, scores, expert_indices)."
            )

        router_logits, router_scores, _selected_experts = output
        allowed = self._expert_subsets[router][self._subset].to(router_logits.device)
        allowed_logits = router_logits.index_select(-1, allowed)
        probabilities = torch.softmax(allowed_logits, dtype=torch.float, dim=-1)
        scores, local_indices = torch.topk(
            probabilities, router_scores.shape[-1], dim=-1
        )
        scores /= scores.sum(dim=-1, keepdim=True)
        selected_experts = allowed[local_indices]
        return router_logits, scores.to(router_logits.dtype), selected_experts

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def rewind_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    crop_parameter = next(iter(inspect.signature(cache.crop).parameters.values()))
    if crop_parameter.name == "tokens_to_remove":
        cache.crop(-1)
    else:
        cache.crop(cache.get_seq_length() - 1)
    return cache


def sample_token(logits: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def main() -> None:
    args = parse_args()
    if args.list_queries:
        for index, query in enumerate(EXAMPLE_QUERIES, start=1):
            print(f"{index:2}. {query}")
        return

    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be nonnegative.")
    if args.temperature < 0:
        raise ValueError("--temperature must be nonnegative.")
    if not 0.0 <= args.expert_overlap < 1.0:
        raise ValueError("--expert-overlap must be in the range [0, 1).")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    use_device_map = args.device_map != "none"
    if use_device_map and not torch.cuda.is_available():
        raise RuntimeError("--device-map requires CUDA; use --device-map none on CPU/MPS.")
    dtype = torch.bfloat16 if use_device_map or device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    load_kwargs: dict[str, Any] = {"dtype": dtype}
    if use_device_map:
        load_kwargs.update(
            device_map=args.device_map,
            max_memory={
                gpu: args.gpu_memory for gpu in range(torch.cuda.device_count())
            },
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if not use_device_map:
        model.to(device)
    model.eval()
    input_device = model.get_input_embeddings().weight.device

    blocks = find_decoder_blocks(model)
    config = RecirculationConfig(
        destination=args.destination,
        source=resolve_source(args.source, len(blocks)),
        alpha=args.alpha,
        beta=args.beta,
    )
    if not 0 <= config.destination < config.source < len(blocks):
        raise ValueError(
            f"The model has {len(blocks)} decoder blocks, but the requested "
            f"indices were destination={config.destination}, source={config.source}."
        )
    expert_shards = (
        TemporalExpertShards(
            blocks[config.destination :], args.seed, args.expert_overlap
        )
        if args.tempshard == 2
        else None
    )
    if expert_shards is not None and not expert_shards.is_moe:
        expert_shards.close()
        expert_shards = None

    def step(token: Tensor, cache: DynamicCache) -> tuple[Tensor, DynamicCache]:
        past_length = cache.get_seq_length()
        token_length = token.shape[1]
        cache_position = torch.arange(
            past_length,
            past_length + token_length,
            device=token.device,
        )
        attention_mask = torch.ones(
            token.shape[0],
            past_length + token_length,
            dtype=torch.long,
            device=token.device,
        )
        outputs = model(
            input_ids=token,
            attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        return outputs.logits, outputs.past_key_values

    prompt = args.prompt or EXAMPLE_QUERIES[args.query_index - 1]
    encoded_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    input_ids = encoded_prompt.input_ids.to(input_device)
    if input_ids.shape[1] == 0:
        raise ValueError("The tokenizer produced an empty prompt.")

    eos_token_ids = model.generation_config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos_token_ids = set(eos_token_ids or [])

    def generate(use_recirculation: bool) -> Tensor:
        torch.manual_seed(args.seed)
        if expert_shards is not None:
            if use_recirculation:
                expert_shards.select(0)
            else:
                expert_shards.disable()
        cache = DynamicCache(config=model.config)
        if use_recirculation:
            cache.activate_past_recording()

        if use_recirculation:
            prompt_logits, cache = recirculate(
                input_ids,
                blocks=blocks,
                cache=cache,
                step=step,
                rewind_one=rewind_dynamic_cache,
                config=config,
                select_expert_subset=expert_shards.select if expert_shards else None,
            )
            next_logits = prompt_logits[:, -1, :]
        else:
            for position in range(input_ids.shape[1]):
                token = input_ids[:, position : position + 1]
                token_logits, cache = step(token, cache)
            next_logits = token_logits[:, -1, :]

        generated_ids = input_ids.clone()
        for _ in range(args.max_new_tokens):
            next_token = sample_token(next_logits, args.temperature)
            generated_ids = torch.cat((generated_ids, next_token), dim=1)

            if next_token.item() in eos_token_ids:
                break

            if use_recirculation:
                token_logits, cache = recirculate(
                    next_token,
                    blocks=blocks,
                    cache=cache,
                    step=step,
                    rewind_one=rewind_dynamic_cache,
                    config=config,
                    select_expert_subset=expert_shards.select if expert_shards else None,
                )
            else:
                token_logits, cache = step(next_token, cache)
            next_logits = token_logits[:, -1, :]

        return generated_ids

    def synchronize_devices() -> None:
        if torch.cuda.is_available():
            for gpu in range(torch.cuda.device_count()):
                torch.cuda.synchronize(gpu)

    def timed_generate(use_recirculation: bool) -> tuple[Tensor, float]:
        synchronize_devices()
        start = time.perf_counter()
        generated_ids = generate(use_recirculation=use_recirculation)
        synchronize_devices()
        return generated_ids, time.perf_counter() - start

    prompt_length = input_ids.shape[1]
    if not args.skip_baseline:
        baseline_ids, baseline_seconds = timed_generate(use_recirculation=False)
        print(f"=== Recirculation OFF ({baseline_seconds:.3f} s) ===")
        print(
            tokenizer.decode(
                baseline_ids[0, prompt_length:], skip_special_tokens=True
            )
        )

    recirculated_ids, recirculated_seconds = timed_generate(use_recirculation=True)
    if expert_shards is not None:
        expert_shards.close()

    heading_prefix = "" if args.skip_baseline else "\n"
    print(f"{heading_prefix}=== Recirculation ON ({recirculated_seconds:.3f} s) ===")
    print(
        tokenizer.decode(
            recirculated_ids[0, prompt_length:], skip_special_tokens=True
        )
    )


if __name__ == "__main__":
    main()