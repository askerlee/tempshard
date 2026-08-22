from __future__ import annotations

import argparse
import inspect
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from recirculation import RecirculationConfig, recirculate


EXAMPLE_QUERIES = (
    "Explain why the sky is blue.",
    "A red ball is in a box. The box is moved into a closet. Where is the red ball now?",
    "Maya gave the book to Liam after he finished the puzzle. Who finished the puzzle?",
    "If the first word is an animal, answer FIRST; if the second word is an animal, answer SECOND. Words: violin tiger.",
    "A train leaves at 9:15 AM and travels for 2 hours and 45 minutes. What time does it arrive?",
    "Fred took his fishing pole to the bank of a river. Is Fred likely to find an ATM at this bank? Explain briefly.",
    "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Explain.",
    "The meeting was moved from Tuesday to Friday, then moved two days earlier. On what day is the meeting?",
    "Plan three concise steps for making tea when the kettle is empty and the mug is dirty.",
    "A store discounts a $80 jacket by 25%, then adds 10% sales tax. What is the final price?",
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
    parser.add_argument("--source", type=int, default=12)
    # alpha: weight of the source residual stream in the convex combination. 
    parser.add_argument("--alpha", type=float, default=0.15)
    # beta: weight of the destination residual stream in the convex combination. 
    # If None, defaults to 1 - alpha.
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Defaults to 1 - alpha.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Use greedy decoding at 0, sampling above 0.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda:1",
        help="Torch device such as cpu, cuda, or cuda:0. Defaults to auto.",
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

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.eval()

    blocks = find_decoder_blocks(model)
    config = RecirculationConfig(
        destination=args.destination,
        source=args.source,
        alpha=args.alpha,
        beta=args.beta,
    )
    if not 0 <= config.destination < config.source < len(blocks):
        raise ValueError(
            f"The model has {len(blocks)} decoder blocks, but the requested "
            f"indices were destination={config.destination}, source={config.source}."
        )

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
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if input_ids.shape[1] == 0:
        raise ValueError("The tokenizer produced an empty prompt.")

    eos_token_ids = model.generation_config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos_token_ids = set(eos_token_ids or [])

    def generate(use_recirculation: bool) -> Tensor:
        torch.manual_seed(args.seed)
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
                )
            else:
                token_logits, cache = step(next_token, cache)
            next_logits = token_logits[:, -1, :]

        return generated_ids

    baseline_ids = generate(use_recirculation=False)
    recirculated_ids = generate(use_recirculation=True)

    print("=== Recirculation OFF ===")
    print(tokenizer.decode(baseline_ids[0], skip_special_tokens=True))
    print("\n=== Recirculation ON ===")
    print(tokenizer.decode(recirculated_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()