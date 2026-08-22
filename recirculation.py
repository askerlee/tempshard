"""Minimal reference for one-step, training-free recirculation.
   Originally implemented by Benhao Huang:
   https://gist.github.com/huskydoge/1ff29693e2172226ec26081f208b19d6

In source-to-destination mode, for every token:
  1. Run a normal cached pass; return its logits and save residuals h_d, h_s.
  2. Rewind the KV cache by one position.
  3. Run the token again, replacing the output of destination block d with
     beta * h_d + alpha * (||h_d|| / ||h_s||) * h_s.
  4. Ignore the second logits and commit the second-pass KV cache.

In layerwise mode, each block from destination through source is immediately
run a second time with its own first output as input. Its first-pass KV entry is
removed before the repeated call, so the cache still advances by one position
per token.

``step(token, cache)``, ``rewind_one(cache)``, and
``rewind_layer(cache, layer_index)`` are model-specific adapters. Block indices
are zero-based outputs, so the mixture is injected as the input to block d + 1.
This is the serial reference, not the paper's serving pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RecirculationConfig:
    destination: int
    source: int
    alpha: float
    beta: float | None = None  # None selects the convex mix: beta = 1 - alpha.
    eps: float = 1e-8
    mode: Literal["source", "layerwise"] = "source"


def _residual(output: Any) -> Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, Tensor):
        raise TypeError("A transformer block must return its residual stream first.")
    return hidden


class _Hooks:
    def __init__(self, blocks: Sequence[nn.Module], cfg: RecirculationConfig) -> None:
        if not 0 <= cfg.destination < cfg.source < len(blocks):
            raise ValueError("Expected 0 <= destination < source < number of blocks.")
        self.cfg = cfg
        self.mode = "off"
        self.h_d: Tensor | None = None
        self.h_s: Tensor | None = None
        self.handles = (
            blocks[cfg.destination].register_forward_hook(self._save_destination),
            blocks[cfg.source].register_forward_hook(self._save_source),
            blocks[cfg.destination + 1].register_forward_pre_hook(self._inject),
        )

    def _save_destination(self, _module: nn.Module, _inputs: tuple, output: Any) -> None:
        if self.mode == "capture":
            self.h_d = _residual(output).detach().clone()

    def _save_source(self, _module: nn.Module, _inputs: tuple, output: Any) -> None:
        if self.mode == "capture":
            self.h_s = _residual(output).detach().clone()

    def _inject(self, _module: nn.Module, inputs: tuple) -> tuple | None:
        if self.mode != "inject":
            return None
        if self.h_d is None or self.h_s is None:
            raise RuntimeError("The first pass did not capture both residual streams.")

        input_device = _residual(inputs).device
        destination = self.h_d.to(device=input_device, dtype=torch.float32)
        source = self.h_s.to(device=input_device, dtype=torch.float32)
        source *= torch.linalg.vector_norm(destination, dim=-1, keepdim=True) / (
            torch.linalg.vector_norm(source, dim=-1, keepdim=True).clamp_min(self.cfg.eps)
        )
        beta = 1.0 - self.cfg.alpha if self.cfg.beta is None else self.cfg.beta
        mixed = (beta * destination + self.cfg.alpha * source).to(inputs[0].dtype)
        return (mixed, *inputs[1:])

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class _LayerwiseSelfLoops:
    def __init__(
        self,
        blocks: Sequence[nn.Module],
        config: RecirculationConfig,
        cache: Any,
        rewind_layer: Callable[[Any, int], Any],
        select_expert_subset: Callable[[int], None] | None,
    ) -> None:
        if not 0 <= config.destination < config.source < len(blocks):
            raise ValueError("Expected 0 <= destination < source < number of blocks.")
        self.cache = cache
        self.rewind_layer = rewind_layer
        self.select_expert_subset = select_expert_subset
        self.replaying = False
        self.handles = tuple(
            block.register_forward_hook(
                self._repeat_layer(layer_index), with_kwargs=True
            )
            for layer_index, block in enumerate(
                blocks[config.destination : config.source + 1],
                start=config.destination,
            )
        )

    def _repeat_layer(self, layer_index: int) -> Callable[..., Any]:
        def repeat(
            module: nn.Module,
            inputs: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> Any:
            if self.replaying:
                return output

            hidden = _residual(output)
            if inputs:
                repeated_inputs = (hidden, *inputs[1:])
                repeated_kwargs = kwargs
            elif "hidden_states" in kwargs:
                repeated_inputs = inputs
                repeated_kwargs = {**kwargs, "hidden_states": hidden}
            else:
                raise TypeError(
                    "A transformer block must receive its residual stream as "
                    "the first argument or as hidden_states."
                )

            self.cache = self.rewind_layer(self.cache, layer_index)
            if self.select_expert_subset is not None:
                self.select_expert_subset(1)
            self.replaying = True
            try:
                return module(*repeated_inputs, **repeated_kwargs)
            finally:
                self.replaying = False
                if self.select_expert_subset is not None:
                    self.select_expert_subset(0)

        return repeat

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def recirculate(
    input_ids: Tensor,
    *,
    blocks: Sequence[nn.Module],
    cache: Any,
    step: Callable[[Tensor, Any], tuple[Tensor, Any]],
    rewind_one: Callable[[Any], Any],
    config: RecirculationConfig,
    select_expert_subset: Callable[[int], None] | None = None,
    rewind_layer: Callable[[Any, int], Any] | None = None,
) -> tuple[Tensor, Any]:
    """Run source-to-destination recirculation or layerwise self-loops."""

    if config.mode == "layerwise":
        if rewind_layer is None:
            raise ValueError("Layerwise mode requires rewind_layer.")
        logits = []
        hooks = _LayerwiseSelfLoops(
            blocks, config, cache, rewind_layer, select_expert_subset
        )
        try:
            for position in range(input_ids.shape[1]):
                token = input_ids[:, position : position + 1]
                if select_expert_subset is not None:
                    select_expert_subset(0)
                token_logits, cache = step(token, hooks.cache)
                hooks.cache = cache
                logits.append(token_logits)
        finally:
            if select_expert_subset is not None:
                select_expert_subset(0)
            hooks.close()
        return torch.cat(logits, dim=1), hooks.cache

    logits = []
    hooks = _Hooks(blocks, config)
    try:
        for position in range(input_ids.shape[1]):
            token = input_ids[:, position : position + 1]

            if select_expert_subset is not None:
                select_expert_subset(0)
            hooks.h_d = hooks.h_s = None
            hooks.mode = "capture"
            first_logits, cache = step(token, cache)
            hooks.mode = "off"
            logits.append(first_logits)

            cache = rewind_one(cache)
            if select_expert_subset is not None:
                select_expert_subset(1)
            hooks.mode = "inject"
            _ignored_logits, cache = step(token, cache)
            hooks.mode = "off"
    finally:
        if select_expert_subset is not None:
            select_expert_subset(0)
        hooks.close()

    return torch.cat(logits, dim=1), cache

