"""Opt-in laptop training speed-ups: compiled decoder layers and CPU-GPU batch splitting.

These are items 3 and 4 of report Section 3.8.2 (Table 4.14). ``cmr-stage-b --compile``
compiles each frozen decoder layer for training passes only. ``--cpu-share K`` runs the last
K micro-batches of each step on a float32 CPU copy in a background thread; its gradients are
added to the GPU ones before clipping. ``--cpu-share auto`` (stored as -1) resets K every
10 steps from the measured time per example on each side. :mod:`cmr.train.bridge_sft` calls
the training step that uses them the "pilot path".
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from cmr.models.bridge import FrozenBridgeVLM


def _layers(model: FrozenBridgeVLM) -> list[torch.nn.Module]:
    """The frozen LLM's decoder layers (Llama / Qwen2 layout), or [] if not found."""
    base = model.llm
    for attr in ("base_model", "model"):  # PEFT wrapper, then the CausalLM's decoder
        while hasattr(base, attr) and not hasattr(base, "layers"):
            nxt = getattr(base, attr)
            if nxt is base:
                break
            base = nxt
    layers = getattr(base, "layers", None)
    return list(layers) if layers is not None else []


class CompiledLayers:
    """torch.compile each decoder layer once; switch the compiled forward on and off.

    Setting ``_compiled_call_impl`` back to None restores the original forward.
    """

    def __init__(
        self, model: FrozenBridgeVLM, recompile_limit: int = 64, backend: str = "inductor"
    ) -> None:
        self.layers = _layers(model)
        if not self.layers:
            raise ValueError("no decoder layers found to compile")
        dyn = torch._dynamo.config
        for name in ("recompile_limit", "cache_size_limit"):  # name differs across torch
            if hasattr(dyn, name):
                setattr(dyn, name, max(int(getattr(dyn, name)), recompile_limit))
        self._compiled: list[Any] = []
        for layer in self.layers:
            layer.compile(dynamic=False, backend=backend)
            self._compiled.append(layer._compiled_call_impl)
            layer._compiled_call_impl = None  # off until a training pass asks for it

    def set(self, on: bool) -> None:
        for layer, fn in zip(self.layers, self._compiled, strict=True):
            layer._compiled_call_impl = fn if on else None

    @contextmanager
    def active(self) -> Iterator[None]:
        """Compiled forward inside the block, the original one after it."""
        self.set(True)
        try:
            yield
        finally:
            self.set(False)


@contextmanager
def maybe(ctx: CompiledLayers | None) -> Iterator[None]:
    """``ctx.active()`` when compiling, else nothing."""
    if ctx is None:
        yield
    else:
        with ctx.active():
            yield


class CpuCoTrainer:
    """A float32 CPU copy of the model that trains some micro-batches in parallel.

    ``run`` starts a thread calling ``work(cpu_model, job)`` per job (each calls backward
    and returns logged losses); ``join`` waits and returns them.
    """

    def __init__(
        self, cpu_model: FrozenBridgeVLM, threads: int, share: int = 3, auto: bool = False
    ) -> None:
        if cpu_model.device.type != "cpu":
            raise ValueError("the co-training copy must live on the CPU")
        self.model = cpu_model
        self.threads = threads
        self.share = share  # micro-batches per step on the CPU (retuned when ``auto``)
        self.auto = auto
        self._per_ex: list[tuple[float, float]] = []  # (GPU, CPU) seconds per example
        self._thread: threading.Thread | None = None
        self._result: list[dict[str, float]] = []
        self._error: BaseException | None = None
        self.seconds = 0.0  # wall time of the last run

    def params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    @torch.no_grad()
    def sync_from(self, model: FrozenBridgeVLM) -> None:
        """Copy the trainable weights of ``model`` (any device) into the CPU copy."""
        src = [p for p in model.parameters() if p.requires_grad]
        dst = self.params()
        if len(src) != len(dst):
            raise ValueError("models have different trainable parameters")
        for s, d in zip(src, dst, strict=True):
            d.copy_(s.detach().to("cpu", d.dtype))

    def run(
        self, jobs: list[Any], work: Callable[[FrozenBridgeVLM, Any], dict[str, float]]
    ) -> None:
        for p in self.params():
            p.grad = None
        self._result, self._error = [], None

        def target() -> None:
            t0 = time.perf_counter()
            try:
                torch.set_num_threads(self.threads)
                self._result = [work(self.model, job) for job in jobs]
            except BaseException as e:
                self._error = e
            self.seconds = time.perf_counter() - t0

        self._thread = threading.Thread(target=target, name="cpu-cotrain", daemon=True)
        self._thread.start()

    def join(self) -> list[dict[str, float]]:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            raise RuntimeError("CPU co-training worker failed") from self._error
        return self._result

    def observe(self, sec_gpu: float, n_gpu: int, sec_cpu: float, n_cpu: int, total: int) -> None:
        """Record one step's timings; with ``auto``, recompute the CPU share every 10 steps.

        With r = CPU / GPU seconds per example, both sides finish together when
        ``share = total / (1 + r)``; r is the median over the last 10 steps, ignoring the
        first 3 (which include compilation).
        """
        if n_gpu and n_cpu:
            self._per_ex.append((sec_gpu / n_gpu, sec_cpu / n_cpu))
        n = len(self._per_ex)
        if not self.auto or n <= 3 or n % 10:
            return
        ratios = sorted(c / g for g, c in self._per_ex[max(3, n - 10) :] if g > 0)
        if ratios:
            r = ratios[len(ratios) // 2]
            self.share = min(total - 1, max(1, round(total / (1 + r))))

    @torch.no_grad()
    def add_grads_to(self, model: FrozenBridgeVLM) -> None:
        """Add the CPU copy's gradients to ``model``'s trainable parameters."""
        dst = [p for p in model.parameters() if p.requires_grad]
        for d, s in zip(dst, self.params(), strict=True):
            if s.grad is None:
                continue
            g = s.grad.to(d.device, d.dtype)
            if d.grad is None:
                d.grad = g
            else:
                d.grad.add_(g)
