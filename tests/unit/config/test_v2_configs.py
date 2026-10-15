"""The real v2 configs load and their learning-rate groups match the v2 modules (tiny stand-ins)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vivlmoe.config import load_config
from vivlmoe.config.loader import to_dict
from vivlmoe.models.factory import build_task_model, load_model_cfg
from vivlmoe.models.freezing import apply_freeze, freeze_paths_for_preset
from vivlmoe.training.optim import resolve_param_groups

REPO = Path(__file__).resolve().parents[3]
V2 = REPO / "configs" / "v2"
V2_TINY = REPO / "configs" / "model" / "v2_tiny.yaml"
GENERATION = ["ktvic_caption", "uitviic_caption", "openvivqa", "vitextvqa", "viocrvqa"]


def test_v2_model_configs() -> None:
    m = load_model_cfg(REPO / "configs" / "model" / "v2.yaml")
    assert (m.vision.kind, m.vision.name, m.vision.image_size) == (
        "radio",
        "nvidia/C-RADIOv4-SO400M",
        512,
    )
    assert (m.text.arch, m.text.name) == ("modernbert", "jhu-clsp/mmBERT-base")
    assert m.contrastive_head == "sigmoid" and m.logit_scale_init == 0.1 and m.moe is None
    s1 = m.stage1
    assert (s1.teacher_kind, s1.teacher_name) == ("siglip", "google/siglip2-giant-opt-patch16-384")
    assert s1.teacher_lowercase and s1.student_pooling == "mean"
    g = m.generation
    assert (g.kind, g.name, g.llm_class) == ("causal_lm", "Qwen/Qwen3.5-0.8B", "Qwen3_5ForCausalLM")
    assert (m.encoder.dim, m.encoder.num_layers, m.resampler.num_latents) == (768, 12, 64)
    for name, size in (("v2_siglip2g", None), ("v2_siglip2_so400m", None)):
        alt = load_model_cfg(REPO / "configs" / "model" / f"{name}.yaml")
        assert alt.vision.kind == "siglip" and alt.vision.image_size == size
        assert alt.generation.kind == "causal_lm"  # everything else from v2.yaml


def test_distill_and_align_configs() -> None:
    d = load_config(V2 / "distill_text.yaml")
    assert d.task.name == "stage1_teacher" and d.task.eval_clip_retrieval
    assert set(d.data.train.options["sources"]) == {"captions", "phomt"}
    assert d.data.tokenizer.en_max_len == 64 and d.eval.selection_metric == "ktvic_dev/rsum"
    assert set(d.data.evals) == {"phomt_dev", "ktvic_dev", "uitviic_dev"}
    no_phomt = load_config(V2 / "distill_text.yaml", ["data.train.options.sources.phomt=null"])
    assert no_phomt.data.train.options["sources"]["phomt"] is None
    a = load_config(V2 / "align.yaml")
    assert a.task.name == "contrastive" and a.optim.freeze_preset == "sparse"
    assert a.init_from is not None and a.init_from.load == ["text_encoder"]
    assert a.init_from.path == "runs/v2_distill_text/checkpoints/best.pt"
    assert a.data.transform.source == "tower" and a.model.contrastive_head == "sigmoid"
    assert (a.data.batch_size, a.train.grad_accum_steps) == (256, 2)  # one A40: 512 per step
    # the groups cover every trainable v2 backbone parameter (strict coverage)
    model = build_task_model("backbone", load_model_cfg(V2_TINY))
    apply_freeze(model, freeze_paths_for_preset(model, a.optim.freeze_preset))
    groups = resolve_param_groups(model, a.optim)
    assert {grp.name.split("/")[0] for grp in groups} == {"text", "fusion"}


def _sidecar(path: Path, moe: dict | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not read")
    model = to_dict(load_model_cfg(V2_TINY))
    model["moe"] = moe
    config = {"model": model, "loss": {"router": {"use_z": True, "use_mi": True, "use_lb": False}}}
    meta = {"format_version": 1, "config": config, "tower_provenance": {}, "step": 0}
    path.with_name(path.name + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


MOE = {"num_shared": 1, "num_routed": 8, "top_k": 2, "aux_free_bias": True}


def test_upcycle_and_sparse_configs(tmp_path: Path) -> None:
    dense = _sidecar(tmp_path / "align.pt", None)
    up = load_config(V2 / "upcycle.yaml", [f"model.from_checkpoint={dense}", "train.max_steps=1"])
    assert up.run.name == "v2_moe_upcycled" and up.model.moe is not None
    assert (up.model.moe.num_shared, up.model.moe.num_routed, up.model.moe.top_k) == (1, 8, 2)
    assert up.model.contrastive_head == "sigmoid"  # inherited from the dense v2 sidecar
    upcycled = _sidecar(tmp_path / "upcycled.pt", MOE)
    sp = load_config(
        V2 / "sparse.yaml", [f"model.from_checkpoint={upcycled}", f"init_from.path={upcycled}"]
    )
    assert sp.run.name == "v2_moe_sparse" and sp.optim.lr == 1.0e-5 and sp.train.max_steps == 6000
    assert sp.data.transform.source == "tower" and sp.optim.freeze_preset == "sparse"
    assert (sp.data.batch_size, sp.train.grad_accum_steps) == (256, 2)


@pytest.mark.parametrize("name", [*GENERATION, "vimacsa"])
def test_downstream_configs(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _sidecar(tmp_path / "runs" / "v2_moe_sparse" / "checkpoints" / "best.pt", MOE)
    cfg = load_config(V2 / "downstream" / f"{name}.yaml")
    assert cfg.run.name == f"v2_{name}" and cfg.checkpoint.keep_last_n == 1
    assert cfg.data.batch_size * cfg.train.grad_accum_steps == 32  # the paper batch on one A40
    assert cfg.model.from_checkpoint == "runs/v2_moe_sparse/checkpoints/best.pt"
    assert cfg.init_from is not None and cfg.init_from.load == ["all"]
    assert cfg.data.transform.source == "tower" and cfg.model.moe is not None
    if name == "vimacsa":
        assert cfg.task.name == "vimacsa" and not cfg.optim.groups
        return
    assert cfg.model.generation.kind == "causal_lm"
    assert [g.name for g in cfg.optim.groups] == ["llm", "projector", "backbone"]
    assert ("{question}" in cfg.task.llm_prompt_template) == (cfg.task.name == "vqa")
    # the groups cover every trainable parameter of the LLM-head model under the task's preset
    model = build_task_model("seq2seq", load_model_cfg(V2_TINY))
    preset = "downstream_caption" if cfg.task.name == "captioning" else "downstream_seq2seq"
    apply_freeze(model, [*freeze_paths_for_preset(model, preset), "backbone.logit_bias"])
    assert {g.name.split("/")[0] for g in resolve_param_groups(model, cfg.optim)} == {
        "llm",
        "projector",
        "backbone",
    }
