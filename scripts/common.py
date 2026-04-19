from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import psutil
import torch
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar as disable_datasets_progress_bar
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

try:
    from huggingface_hub.utils import disable_progress_bars as disable_hf_hub_progress_bars
except ImportError:
    disable_hf_hub_progress_bars = None

from kvpress import ExpectedAttentionPress

try:
    import multiprocess.resource_tracker as multiprocess_resource_tracker
except ImportError:
    multiprocess_resource_tracker = None


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_MODEL_CANDIDATES = (
    "Qwen/Qwen2.5-0.5B",
    "HuggingFaceTB/SmolLM2-360M",
)
DEFAULT_TASK1_RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
DEFAULT_TASK23_RATIOS = (0.2, 0.4, 0.6, 0.8)
DEFAULT_DATASET_NAME = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"


def patch_resource_tracker_teardown_bug() -> None:
    if multiprocess_resource_tracker is None:
        return

    tracker_cls = multiprocess_resource_tracker.ResourceTracker
    if getattr(tracker_cls, "_kvpress_demo_patched", False):
        return

    original_del = tracker_cls.__del__

    def safe_del(self: Any) -> None:
        try:
            original_del(self)
        except AttributeError as exc:
            if "_recursion_count" not in str(exc):
                raise

    tracker_cls.__del__ = safe_del
    tracker_cls._kvpress_demo_patched = True


patch_resource_tracker_teardown_bug()
transformers_logging.disable_progress_bar()
transformers_logging.set_verbosity_error()
disable_datasets_progress_bar()
if disable_hf_hub_progress_bars is not None:
    disable_hf_hub_progress_bars()


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    model_name: str
    device: str
    dtype: torch.dtype
    notes: list[str]


@dataclass
class SampleRecord:
    input_id: int
    source_index: int
    text: str
    input_ids: torch.Tensor

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[-1])


class MemoryTracker:
    def __init__(self, device: str):
        self.device = device
        self.values_mb: list[float] = []
        self.metric_name = self._detect_metric_name()
        gc.collect()
        if self.device == "mps":
            try:
                torch.mps.empty_cache()
                synchronize_device(self.device)
            except Exception:
                pass
        self.baseline_mb = current_memory_mb(self.device)

    def _detect_metric_name(self) -> str:
        if self.device == "mps":
            has_current = hasattr(torch.mps, "current_allocated_memory")
            has_driver = hasattr(torch.mps, "driver_allocated_memory")
            if has_current and has_driver:
                return "mps_driver_or_current_allocated_mb_delta"
            if has_driver:
                return "mps_driver_allocated_mb_delta"
            if has_current:
                return "mps_current_allocated_mb_delta"
        return "process_rss_mb_delta"

    def sample(self) -> float:
        mb = max(current_memory_mb(self.device) - self.baseline_mb, 0.0)
        self.values_mb.append(mb)
        return mb

    def peak_mb(self) -> float:
        if not self.values_mb:
            self.sample()
        return max(self.values_mb)


def current_memory_mb(device: str) -> float:
    if device == "mps":
        values: list[float] = []
        if hasattr(torch.mps, "current_allocated_memory"):
            values.append(float(torch.mps.current_allocated_memory()) / (1024**2))
        if hasattr(torch.mps, "driver_allocated_memory"):
            values.append(float(torch.mps.driver_allocated_memory()) / (1024**2))
        if values:
            return max(values)
    rss = psutil.Process(os.getpid()).memory_info().rss
    return float(rss) / (1024**2)


def detect_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def device_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "mps" else torch.float32


def synchronize_device(device: str) -> None:
    if device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def cleanup_torch(model: Any | None = None) -> None:
    if model is not None:
        del model
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            synchronize_device("mps")
        except Exception:
            pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def parse_ratio_list(raw: str) -> list[float]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    return values


def format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def load_model_bundle(
    preferred_device: str | None = None,
    model_candidates: Sequence[str] | None = None,
    extra_notes: Sequence[str] | None = None,
) -> ModelBundle:
    preferred_device = preferred_device or detect_device()
    model_candidates = tuple(model_candidates or DEFAULT_MODEL_CANDIDATES)
    devices_to_try = [preferred_device]
    if preferred_device == "mps":
        devices_to_try.append("cpu")

    notes = list(extra_notes or [])
    errors: list[str] = []
    tokenizer_cache: dict[str, Any] = {}

    for device in devices_to_try:
        dtype = device_dtype(device)
        for model_name in model_candidates:
            model = None
            try:
                tokenizer = tokenizer_cache.get(model_name)
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    tokenizer_cache[model_name] = tokenizer
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=dtype,
                    attn_implementation="eager",
                    low_cpu_mem_usage=True,
                )
                model.to(device)
                model.eval()
                if getattr(model.generation_config, "pad_token_id", None) is None:
                    model.generation_config.pad_token_id = tokenizer.pad_token_id
                if device != preferred_device:
                    notes.append(f"Loaded {model_name} on {device} after {preferred_device} fallback.")
                return ModelBundle(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    device=device,
                    dtype=dtype,
                    notes=notes,
                )
            except Exception as exc:
                errors.append(f"{model_name} on {device}: {format_exception(exc)}")
                cleanup_torch(model)

    raise RuntimeError("Unable to load any candidate model.\n" + "\n".join(errors))


def load_wikitext_samples(
    tokenizer: Any,
    split: str,
    num_samples: int,
    seq_len_cap: int,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
) -> list[SampleRecord]:
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    rows: list[SampleRecord] = []
    input_id = 0
    for source_index, row in enumerate(dataset):
        text = row["text"].strip()
        if not text:
            continue
        token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[:, :seq_len_cap]
        if token_ids.numel() == 0:
            continue
        rows.append(
            SampleRecord(
                input_id=input_id,
                source_index=source_index,
                text=text,
                input_ids=token_ids,
            )
        )
        input_id += 1
        if len(rows) >= num_samples:
            break
    return rows


def split_tail_window(input_ids: torch.Tensor, eval_length: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    total_length = int(input_ids.shape[-1])
    if total_length <= eval_length:
        return None
    prefix = input_ids[:, : total_length - eval_length]
    eval_ids = input_ids[:, total_length - eval_length :]
    return prefix, eval_ids


def split_prefix_eval_window(
    input_ids: torch.Tensor,
    context_length: int,
    eval_length: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    total_needed = context_length + eval_length
    if int(input_ids.shape[-1]) < total_needed:
        return None
    prefix = input_ids[:, :context_length]
    eval_ids = input_ids[:, context_length:total_needed]
    return prefix, eval_ids


def logits_to_token_nll(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return -gathered


def mean_kl_divergence(reference_logits: torch.Tensor, approx_logits: torch.Tensor) -> float:
    reference_log_probs = torch.log_softmax(reference_logits.float(), dim=-1)
    approx_log_probs = torch.log_softmax(approx_logits.float(), dim=-1)
    reference_probs = reference_log_probs.exp()
    kl = torch.sum(reference_probs * (reference_log_probs - approx_log_probs), dim=-1)
    return float(kl.mean().item())


def compute_mean_hidden_state_norm(hidden_state: torch.Tensor) -> float:
    return float(hidden_state.float().norm(dim=-1).mean().item())


def compute_attention_entropy(attention_tensor: torch.Tensor | None) -> float:
    if attention_tensor is None:
        return math.nan
    probs = attention_tensor.float().clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    return float(entropy.item())


def build_expected_attention_press(compression_ratio: float, n_future_positions: int) -> ExpectedAttentionPress:
    return ExpectedAttentionPress(
        compression_ratio=compression_ratio,
        n_future_positions=n_future_positions,
    )


def evaluate_window(
    model: Any,
    prefix_ids: torch.Tensor,
    eval_ids: torch.Tensor,
    device: str,
    press: ExpectedAttentionPress | None = None,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
    memory_tracker: MemoryTracker | None = None,
) -> dict[str, Any]:
    if prefix_ids.shape[0] != 1 or eval_ids.shape[0] != 1:
        raise ValueError("Expected batch size 1 inputs.")

    context = press(model) if press is not None else nullcontext()
    synchronize_device(device)
    started_at = time.perf_counter()
    with torch.no_grad(), context:
        prefix_outputs = model(
            prefix_ids,
            use_cache=True,
            return_dict=True,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        if memory_tracker is not None:
            memory_tracker.sample()
        first_logits = prefix_outputs.logits[:, -1:, :]
        cache = prefix_outputs.past_key_values
    continuation_logits = None
    if eval_ids.shape[1] > 1:
        with torch.no_grad():
            continuation_outputs = model(
                eval_ids[:, :-1],
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            continuation_logits = continuation_outputs.logits
            if memory_tracker is not None:
                memory_tracker.sample()
    synchronize_device(device)
    elapsed = time.perf_counter() - started_at

    logits = first_logits.float()
    if continuation_logits is not None:
        logits = torch.cat([logits, continuation_logits.float()], dim=1)

    final_hidden_state = None
    if output_hidden_states and prefix_outputs.hidden_states:
        final_hidden_state = prefix_outputs.hidden_states[-1]
    final_attention = None
    if output_attentions and prefix_outputs.attentions:
        final_attention = prefix_outputs.attentions[-1]

    processed_tokens = int(prefix_ids.shape[1] + max(0, eval_ids.shape[1] - 1))
    return {
        "logits": logits,
        "elapsed_sec": elapsed,
        "processed_tokens": processed_tokens,
        "mean_hidden_state_norm": compute_mean_hidden_state_norm(final_hidden_state) if final_hidden_state is not None else math.nan,
        "attention_entropy": compute_attention_entropy(final_attention),
    }


def maybe_fallback_bundle_for_press(
    bundle: ModelBundle,
    prefix_ids: torch.Tensor,
    eval_ids: torch.Tensor,
    compression_ratio: float,
    n_future_positions: int,
) -> ModelBundle:
    if bundle.device != "mps":
        return bundle

    press = build_expected_attention_press(compression_ratio=compression_ratio, n_future_positions=n_future_positions)
    try:
        evaluate_window(bundle.model, prefix_ids, eval_ids, device=bundle.device, press=press)
        return bundle
    except Exception as exc:
        note = (
            "ExpectedAttentionPress failed on MPS and the experiment switched to CPU fallback. "
            f"Root cause: {format_exception(exc)}"
        )
        cleanup_torch(bundle.model)
        model_candidates = (bundle.model_name,) + tuple(name for name in DEFAULT_MODEL_CANDIDATES if name != bundle.model_name)
        return load_model_bundle(preferred_device="cpu", model_candidates=model_candidates, extra_notes=bundle.notes + [note])


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def summarize_notes(notes: Iterable[str]) -> None:
    for note in notes:
        print(f"[note] {note}")
