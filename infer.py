from __future__ import annotations

import argparse
import inspect
import sys
import time
from collections.abc import Sequence
from itertools import combinations
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from recirculation import MagnitudeDiffStats, RecirculationConfig, recirculate


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
    "A coastal town must decide whether to rebuild a storm-damaged seawall, restore wetlands, or relocate the most exposed homes. Compare the options across cost, resilience, fairness, and uncertainty, then propose a decision process that can adapt as conditions change.",
    "A hospital has fewer intensive-care beds than patients likely to need them during an outbreak. Design a transparent allocation policy, explain how it handles changing prognoses and ties, and identify safeguards against bias and avoidable harm.",
    "Two departments report conflicting results from the same customer survey: one says satisfaction improved, while the other says complaints became more severe. Explain how both claims could be true and outline an analysis that would reconcile the evidence.",
    "A teacher discovers that students are using generative AI for homework, but the school has no clear policy. Develop a response that supports learning, treats students fairly, and distinguishes acceptable assistance from work that misrepresents understanding.",
    "An old bridge is still considered safe but requires increasingly frequent repairs. Compare continued maintenance, major rehabilitation, and replacement while accounting for disruption, uncertain future demand, public safety, and budget constraints.",
    "A neighborhood wants more housing but disagrees about building height, affordability requirements, parking, and preservation of local businesses. Propose a negotiation framework and a compromise plan, including who bears each cost and how outcomes should be measured.",
    "A research team finds a statistically significant effect that is much smaller than expected and disappears under one reasonable analysis choice. Interpret the result, identify what should be reported, and recommend the next study without reducing the decision to a single p-value.",
    "A family must choose between caring for an aging relative at home, hiring in-home support, or moving them to assisted living. Build a respectful decision process that considers autonomy, safety, finances, caregiver capacity, and how the plan should be revisited over time.",
    "A news platform wants to reduce misinformation without suppressing legitimate disagreement or breaking-news updates that later change. Design a moderation approach that combines labels, distribution rules, appeals, and evidence standards, then explain its likely failure modes.",
    "A manufacturer can lower emissions by replacing equipment now, purchasing cleaner electricity, or waiting for a promising technology still under development. Recommend a staged strategy using plausible assumptions about cost, risk, and regulation, and specify signals that would trigger a change in course.",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Generate text with multi-pass residual-stream recirculation."
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
        default=-4,
        help="Source block index; -n selects the nth-to-last block (default: -1).",
    )
    parser.add_argument(
        "--mode",
        choices=("source", "layerwise"),
        default="source",
        help="Recirculate source features or repeat each selected layer in place.",
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
    parser.add_argument(
        "--ortho-mix",
        action="store_true",
        help="Remove the source component parallel to the destination before mixing.",
    )
    parser.add_argument(
        "--ortho_mix_coeff",
        type=float,
        default=0.1,
        help="Projection-removal coefficient used by --ortho-mix (default: 0.1).",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=2,
        help="Total model passes per token, including the initial pass (default: 2).",
    )
    parser.add_argument(
        "--exp_emb",
        action="store_true",
        help="Subtract the top-K expected token embedding from the source latent.",
    )
    parser.add_argument(
        "--exp_emb_K",
        type=int,
        default=1,
        metavar="K",
        help="Number of tokens used for the expected embedding (default: 1).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument(
        "--ablations",
        action="store_true",
        help=(
            "Arguments before this option form the baseline. Each argument after "
            "it creates a separate run overriding only that argument. With no "
            "following arguments, run the baseline only."
        ),
    )
    parser.add_argument(
        "--tempshard",
        action="store_true",
        help="Split MoE experts into one temporal shard per pass.",
    )
    parser.add_argument(
        "--expert-overlap",
        type=float,
        default=0.2,
        help="Fraction of total experts assigned to each pair's common group (default: 0.2).",
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
    if "--ablations" not in argv:
        args = parser.parse_args(argv)
        args.ablations = False
        return args

    ablations_index = argv.index("--ablations")
    baseline_argv = argv[:ablations_index]
    ablations_argv = argv[ablations_index + 1 :]
    args = parser.parse_args(baseline_argv)
    if not ablations_argv:
        args.ablations = None
        return args

    ablations: list[tuple[str, Any]] = []
    index = 0
    while index < len(ablations_argv):
        option = ablations_argv[index]
        option_name, separator, _value = option.partition("=")
        action = parser._option_string_actions.get(option_name)
        if action is None or action.dest == "ablations":
            parser.error(f"invalid ablation argument: {option_name}")

        argument_group = [option]
        if not separator and action.nargs != 0:
            if action.nargs is None:
                value_count = 1
            elif isinstance(action.nargs, int):
                value_count = action.nargs
            else:
                parser.error(
                    f"ablation does not support variable-length argument {option_name}"
                )
            argument_group.extend(
                ablations_argv[index + 1 : index + 1 + value_count]
            )
            if len(argument_group) != value_count + 1:
                parser.error(f"argument {option_name} expected {value_count} value(s)")
            index += value_count

        variation = parser.parse_args([*baseline_argv, *argument_group])
        is_boolean_option = action.nargs == 0 and isinstance(action.const, bool)
        ablation_value = (
            not getattr(args, action.dest)
            if is_boolean_option
            else getattr(variation, action.dest)
        )
        ablations.append((action.dest, ablation_value))
        index += 1

    args.ablations = tuple(ablations)
    return args


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_source(source: int, num_blocks: int) -> int:
    return num_blocks + source if source < 0 else source


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


def resolve_pairwise_expert_count(
    num_experts: int, overlap: float, num_shards: int
) -> int:
    if num_shards < 2:
        return 0
    return min(round(num_experts * overlap), num_experts)


class TemporalExpertShards:
    def __init__(
        self,
        blocks: Sequence[nn.Module],
        num_shards: int,
        seed: int,
        overlap: float = 0.2,
    ) -> None:
        if num_shards < 1:
            raise ValueError("The number of temporal shards must be at least 1.")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("Expert overlap must be in the range [0, 1).")
        self._num_shards = num_shards
        self._enabled = False
        self._subset = 0
        self._expert_subsets: dict[nn.Module, tuple[Tensor, ...]] = {}
        self._handles: list[Any] = []
        layout_counts: dict[
            tuple[int, int, tuple[int, ...], tuple[int, ...]], int
        ] = {}
        partitions: dict[int, tuple[Tensor, ...]] = {}
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
            pairwise_count = resolve_pairwise_expert_count(
                num_experts, overlap, num_shards
            )
            shard_pairs = tuple(combinations(range(num_shards), 2))
            subsets = partitions.get(num_experts)
            if subsets is None:
                pairwise_experts = {
                    pair: torch.randperm(num_experts, generator=generator)[
                        :pairwise_count
                    ]
                    for pair in shard_pairs
                }
                shared_mask = torch.zeros(num_experts, dtype=torch.bool)
                for shared in pairwise_experts.values():
                    shared_mask[shared] = True
                exclusive = torch.arange(num_experts)[~shared_mask]
                exclusive_subsets = torch.tensor_split(
                    exclusive[torch.randperm(len(exclusive), generator=generator)],
                    num_shards,
                )
                subsets = tuple(
                    torch.unique(
                        torch.cat(
                            [
                                exclusive_subsets[subset],
                                *(
                                    pairwise_experts[pair]
                                    for pair in shard_pairs
                                    if subset in pair
                                ),
                            ]
                        )
                    )
                    for subset in range(num_shards)
                )
                partitions[num_experts] = subsets
            minimum_shard_size = min(len(subset) for subset in subsets)
            if top_k > minimum_shard_size:
                raise ValueError(
                    f"The router selects {top_k} experts, but the smallest temporal "
                    f"shard contains only {minimum_shard_size}."
                )
            self._expert_subsets[router] = subsets
            self._handles.append(router.register_forward_hook(self._route_with_subset))
            layout = (
                num_experts,
                pairwise_count,
                tuple(len(subset) for subset in subsets),
                tuple(
                    int(torch.isin(subsets[left], subsets[right]).sum())
                    for left, right in shard_pairs
                ),
            )
            layout_counts[layout] = layout_counts.get(layout, 0) + 1

        print(
            f"Temporal expert sharding into {num_shards} sets with "
            f"{overlap:.0%} overlap for {len(self._handles)} MoE layers."
        )
        shard_pairs = tuple(combinations(range(num_shards), 2))
        for (
            total_experts,
            common_group_size,
            shard_sizes,
            pairwise_shared,
        ), layer_count in layout_counts.items():
            shared_by_pair = {
                f"{left}-{right}": count
                for (left, right), count in zip(shard_pairs, pairwise_shared)
            }
            print(
                f"  {layer_count} layer(s): {total_experts} total experts; "
                f"{common_group_size} assigned to each pair group; "
                f"experts per shard {list(shard_sizes)}; "
                f"actual intersections {shared_by_pair}."
            )

    @property
    def is_moe(self) -> bool:
        return bool(self._handles)

    def select(self, subset: int) -> None:
        if not 0 <= subset < self._num_shards:
            raise ValueError(
                f"Temporal expert subset must be in [0, {self._num_shards})."
            )
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


def rewind_dynamic_cache_layer(
    cache: DynamicCache, layer_index: int
) -> DynamicCache:
    layer = cache.layers[layer_index]
    crop_parameter = next(iter(inspect.signature(layer.crop).parameters.values()))
    if crop_parameter.name == "tokens_to_remove":
        layer.crop(-1)
    else:
        layer.crop(layer.get_seq_length() - 1)
    return cache


def sample_token(logits: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def format_run_arguments(args: argparse.Namespace, options: Sequence[str]) -> str:
    return ", ".join(
        f"{option.replace('_', '-')}={getattr(args, option)}" for option in options
    )


def main() -> None:
    args = parse_args()
    if args.list_queries:
        for index, query in enumerate(EXAMPLE_QUERIES, start=1):
            print(f"{index:2}. {query}")
        return

    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be nonnegative.")
    if args.passes < 1:
        raise ValueError("--passes must be at least 1.")
    if args.exp_emb_K < 1:
        raise ValueError("--exp_emb_K must be at least 1.")
    if args.exp_emb and (
        args.mode != "source" or args.source != -1 or args.destination != 0
    ):
        print(
            "--exp_emb overrides "
            f"--mode {args.mode} --source {args.source} --destination {args.destination} "
            "with --mode source --source -1 --destination 0."
        )
        args.mode = "source"
        #args.source = -1
        #args.destination = 0

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

    def expected_embedding(logits: Tensor, requested_top_k: int) -> Tensor:
        embedding = model.get_input_embeddings()
        top_k = min(requested_top_k, logits.shape[-1])
        top_logits, token_ids = torch.topk(logits, top_k, dim=-1)
        probabilities = torch.softmax(top_logits, dtype=torch.float32, dim=-1)
        token_embeddings = embedding(token_ids.to(embedding.weight.device))
        return torch.sum(
            probabilities.to(token_embeddings.device).unsqueeze(-1)
            * token_embeddings.to(dtype=torch.float32),
            dim=-2,
        ).to(dtype=token_embeddings.dtype)

    blocks = find_decoder_blocks(model)

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

    def generate(
        use_recirculation: bool,
        run_args: argparse.Namespace,
        run_config: RecirculationConfig,
        expert_shards: TemporalExpertShards | None,
    ) -> Tensor:
        torch.manual_seed(run_args.seed)
        magnitude_diff_stats = MagnitudeDiffStats()
        use_expert_shards = expert_shards is not None and run_args.tempshard
        if expert_shards is not None:
            if use_recirculation and use_expert_shards:
                expert_shards.select(0)
            else:
                expert_shards.disable()
        cache = DynamicCache(config=model.config)
        if use_recirculation:
            cache.activate_past_recording()
        expected_embedding_fn = (
            (lambda logits: expected_embedding(logits, run_args.exp_emb_K))
            if run_args.exp_emb
            else None
        )

        if use_recirculation:
            prompt_logits, cache = recirculate(
                input_ids,
                blocks=blocks,
                cache=cache,
                step=step,
                rewind_one=rewind_dynamic_cache,
                config=run_config,
                select_expert_subset=expert_shards.select if use_expert_shards else None,
                expected_embedding=expected_embedding_fn,
                magnitude_diff_stats=magnitude_diff_stats,
                passes=run_args.passes,
                rewind_layer=rewind_dynamic_cache_layer,
            )
            next_logits = prompt_logits[:, -1, :]
        else:
            for position in range(input_ids.shape[1]):
                token = input_ids[:, position : position + 1]
                token_logits, cache = step(token, cache)
            next_logits = token_logits[:, -1, :]

        generated_ids = input_ids.clone()
        for _ in range(run_args.max_new_tokens):
            next_token = sample_token(next_logits, run_args.temperature)
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
                    config=run_config,
                    select_expert_subset=expert_shards.select if use_expert_shards else None,
                    expected_embedding=expected_embedding_fn,
                    magnitude_diff_stats=magnitude_diff_stats,
                    passes=run_args.passes,
                    rewind_layer=rewind_dynamic_cache_layer,
                )
            else:
                token_logits, cache = step(next_token, cache)
            next_logits = token_logits[:, -1, :]

        if use_recirculation and magnitude_diff_stats.mean is not None:
            print(f"magnitude_diff_stats.mean = {magnitude_diff_stats.mean:.3f}")
            
        return generated_ids

    def synchronize_devices() -> None:
        if torch.cuda.is_available():
            for gpu in range(torch.cuda.device_count()):
                torch.cuda.synchronize(gpu)

    def timed_generate(
        use_recirculation: bool, run_args: argparse.Namespace
    ) -> tuple[Tensor, float]:
        run_config = RecirculationConfig(
            destination=run_args.destination,
            source=resolve_source(run_args.source, len(blocks)),
            alpha=run_args.alpha,
            beta=run_args.beta,
            ortho_mix=run_args.ortho_mix,
            ortho_mix_coeff=run_args.ortho_mix_coeff,
            mode=run_args.mode,
        )
        if not 0 <= run_config.destination < run_config.source < len(blocks):
            raise ValueError(
                f"The model has {len(blocks)} decoder blocks, but the requested "
                f"indices were destination={run_config.destination}, "
                f"source={run_config.source}."
            )
        sharded_blocks = (
            blocks[run_config.destination : run_config.source + 1]
            if run_config.mode == "layerwise"
            else blocks[run_config.destination :]
        )
        expert_shards = (
            TemporalExpertShards(
                sharded_blocks,
                num_shards=run_args.passes,
                seed=run_args.seed,
                overlap=run_args.expert_overlap,
            )
            if use_recirculation and run_args.tempshard
            else None
        )
        if expert_shards is not None and not expert_shards.is_moe:
            expert_shards.close()
            expert_shards = None
        synchronize_devices()
        start = time.perf_counter()
        try:
            generated_ids = generate(
                use_recirculation=use_recirculation,
                run_args=run_args,
                run_config=run_config,
                expert_shards=expert_shards,
            )
            synchronize_devices()
            return generated_ids, time.perf_counter() - start
        finally:
            if expert_shards is not None:
                expert_shards.close()

    prompt_length = input_ids.shape[1]
    if args.ablations is False:
        runs = [
            (args, False, "Recirculation OFF"),
            (args, True, "Recirculation ON"),
        ]
    elif args.ablations is None:
        runs = [(args, True, "Recirculation ON")]
    else:
        ablated_options = tuple(dict.fromkeys(option for option, _ in args.ablations))
        baseline_arguments = format_run_arguments(args, ablated_options)
        runs = [(args, True, f"Baseline: {baseline_arguments}")]
        for option, value in args.ablations:
            ablation_args = argparse.Namespace(**vars(args))
            setattr(ablation_args, option, value)
            arguments = format_run_arguments(ablation_args, ablated_options)
            runs.append((ablation_args, True, f"Ablation: {arguments}"))

    for run_index, (run_args, use_recirculation, label) in enumerate(runs):
        run_ids, run_seconds = timed_generate(
            use_recirculation=use_recirculation, run_args=run_args
        )
        if run_index == 0:
            print("\n=== Query ===")
            print(prompt)
        print(f"\n=== {label} ({run_seconds:.3f} s) ===")
        print(
            tokenizer.decode(
                run_ids[0, prompt_length:], skip_special_tokens=True
            )
        )


if __name__ == "__main__":
    main()