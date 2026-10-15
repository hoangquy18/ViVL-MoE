"""Load the decoder-only LLM of the v2 generation head (``model.generation.kind: causal_lm``).

Qwen3.5 checkpoints are composite (vision + language). Loading them through ``AutoModelForCausalLM``
maps to the flat text class and silently re-initialises the language model (transformers#49324), so
the concrete class is named explicitly (``llm_class``, e.g. ``"Qwen3_5ForCausalLM"``) and **any missing
weight is an error**. Trainable weights are fp32 (``load_fp32``); AMP picks the compute dtype.

Tiny random models for CPU tests use Qwen3 or Qwen3-Next (hybrid linear attention, like Qwen3.5),
both shipped with transformers 4.57.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import transformers
from transformers import AutoModelForCausalLM, PreTrainedModel

from vivlmoe.utils.hf import resolve_model_path

log = logging.getLogger(__name__)

TinyArch = Literal["qwen3", "qwen3_next"]
_TINY_CLASSES: dict[str, tuple[str, str]] = {
    "qwen3": ("Qwen3Config", "Qwen3ForCausalLM"),
    "qwen3_next": ("Qwen3NextConfig", "Qwen3NextForCausalLM"),
}


class MissingWeightsError(RuntimeError):
    """The checkpoint did not provide every weight of the LLM class (it would train from random)."""


def _transformers_class(name: str) -> Any:
    cls = getattr(transformers, name, None)
    if cls is None:
        raise ImportError(
            f"transformers {transformers.__version__} has no {name}; the v2 LLM head needs the "
            "environment from requirements-v2.txt (transformers>=5.2 for Qwen3.5)"
        )
    return cls


def _tied_keys(model: PreTrainedModel) -> set[str]:
    """Weight names that may be absent from a checkpoint because they are tied to the embeddings."""
    if not getattr(model.config, "tie_word_embeddings", False):
        return set()
    tied = getattr(model, "_tied_weights_keys", None) or []
    return set(tied.keys() if isinstance(tied, dict) else tied)


def load_causal_lm(
    name: str | None,
    *,
    llm_class: str | None = None,
    revision: str | None = None,
    tiny_config: dict[str, Any] | None = None,
    tiny_arch: TinyArch = "qwen3",
) -> PreTrainedModel:
    """Random tiny model when ``tiny_config`` is given, else ``llm_class.from_pretrained(name)``."""
    if tiny_config is not None:
        cfg_name, model_name = _TINY_CLASSES[tiny_arch]
        model = _transformers_class(model_name)(_transformers_class(cfg_name)(**tiny_config))
        log.info("LLM head: random tiny %s", model_name)
        return model
    if not name:
        raise ValueError("model.generation.name is required for a pretrained causal LM")
    cls = _transformers_class(llm_class) if llm_class else AutoModelForCausalLM
    model, info = cls.from_pretrained(
        resolve_model_path(name), revision=revision, output_loading_info=True
    )
    missing = sorted(set(info.get("missing_keys", [])) - _tied_keys(model))
    if missing:
        raise MissingWeightsError(
            f"{name} loaded as {type(model).__name__} is missing {len(missing)} weights "
            f"(e.g. {missing[:3]}): they would train from random. Set model.generation.llm_class to "
            "the checkpoint's text class (Qwen3.5: Qwen3_5ForCausalLM)"
        )
    log.info("LLM head: %s as %s", name, type(model).__name__)
    return model.float()


__all__ = ["MissingWeightsError", "TinyArch", "load_causal_lm"]
