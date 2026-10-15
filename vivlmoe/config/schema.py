"""Every configuration key of ViVL-MoE, as validated dataclasses.

Only :mod:`vivlmoe.config.loader` knows about YAML; this module knows nothing about torch.
Field comments double as documentation (``docs/configs.md``). Paper references: Sec. 4 (dense
pretraining), Sec. 5 (MoE), Sec. 6.1 (implementation details).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


class ConfigError(ValueError):
    """Raised for unknown keys, type mismatches and cross-field validation failures."""


# --------------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------------
@dataclass
class RunCfg:
    name: str = "run"  # run directory = <dir>/<name>
    dir: str = "runs"
    seed: int = 42  # identical init on every rank; dropout streams are offset per rank
    notes: str = ""


# --------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------
@dataclass
class VisionCfg:
    name: str = (
        "apple/DFN5B-CLIP-ViT-H-14-378"  # HF id (resolved through configs/_base/model_paths.yaml)
    )
    pretrained: bool = True  # False -> random init from `tiny_config` (tests only)
    # random tiny model kwargs when pretrained=False: CLIPVisionConfig / SiglipVisionConfig / RadioStub
    tiny_config: dict[str, Any] | None = None
    kind: Literal["clip", "siglip", "radio"] = "clip"  # v2: SigLIP 2 or C-RADIOv4 (frozen-only)
    image_size: int | None = None  # input resolution; None -> the checkpoint's (C-RADIOv4: 512)
    revision: str | None = None  # pin a hub revision (C-RADIOv4 runs remote code)
    apply_post_layernorm: bool = False  # paper uses V = f_v(I) (raw last hidden state)
    frozen_dtype: Literal["auto", "fp32", "bf16"] = (
        "auto"  # auto: bf16 when frozen on CUDA, else fp32
    )
    gradient_checkpointing: bool = False  # needed for Stage 3 (ViT-H @378 trainable)


@dataclass
class TextCfg:
    name: str = "FacebookAI/xlm-roberta-large"
    pretrained: bool = True
    tiny_config: dict[str, Any] | None = None  # random tiny model config when pretrained=False
    # class of the random tiny model (a pretrained model is built from its own config)
    arch: Literal["xlm-roberta", "modernbert"] = "xlm-roberta"
    gradient_checkpointing: bool = False


@dataclass
class ResamplerCfg:
    """Perceiver resampler R_v (Sec. 5.2): 2 layers, 8 heads x 64, 64 learned queries."""

    num_latents: int = 64
    depth: int = 2
    heads: int = 8
    dim_head: int = 64  # inner attention width = heads * dim_head (512), token width = encoder.dim
    ff_mult: int = 4
    dropout: float = 0.1
    concat_latents_to_kv: bool = (
        False  # Flamingo variant; paper: queries attend to patch tokens only
    )


@dataclass
class EncoderCfg:
    """Shared multimodal encoder f_enc: 12 post-LN Transformer layers (Fig. 3)."""

    dim: int = 768
    num_layers: int = 12
    heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1  # residual dropout after attention / FFN
    attn_dropout: float = 0.1
    ffn_inner_dropout: float = (
        0.1  # inside DenseFFN (act -> dropout -> fc2); separates identical shared experts
    )
    ln_eps: float = 1.0e-5


@dataclass
class MoECfg:
    """Sparse upcycling topology and routing (Sec. 5.3-5.5)."""

    num_shared: int = 1  # N_s always-active experts (0 allowed: output = half the routed path)
    num_routed: int = 8  # N_r routed experts
    top_k: int = 2  # k experts per token (must be 1 when num_routed == 1)
    aux_free_bias: bool = True  # DeepSeek-V3 bias b added to logits at selection only (Eq. 21)
    bias_update_rate: float = 1.0e-3  # eta in Eq. 22
    bias_update_every: Literal["step", "micro"] = (
        "step"  # once per optimizer step (all-reduced) or per micro-batch
    )
    router_init_std: float = 0.02

    def __post_init__(self) -> None:
        if not 0 <= self.num_shared <= 8:
            raise ConfigError(f"model.moe.num_shared must be in [0, 8], got {self.num_shared}")
        if self.num_routed < 1:
            raise ConfigError(f"model.moe.num_routed must be >= 1, got {self.num_routed}")
        if not 1 <= self.top_k <= self.num_routed:
            raise ConfigError(
                f"model.moe.top_k must be in [1, num_routed={self.num_routed}], got {self.top_k} "
                "(paper: k=1 when N_r=1; set top_k explicitly)"
            )
        if self.bias_update_rate < 0:
            raise ConfigError("model.moe.bias_update_rate must be >= 0")


@dataclass
class Stage1Cfg:
    """Stage 1 teacher learning (Sec. 4.1, Eq. 12)."""

    teacher_target: Literal["projected", "eos_hidden"] = (
        "projected"  # CLIP joint space (Appendix A) vs EOS hidden state
    )
    loss_preset: Literal["eq12", "legacy"] = (
        "eq12"  # eq12: mean_i ||z_vi - z_en||^2 ; legacy: normalize + F.mse_loss
    )
    fc1_hidden: int | None = None  # optional hidden width for a 2-layer FC1 (paper: single Linear)
    teacher_tiny_config: dict[str, Any] | None = (
        None  # CLIPTextConfig overrides when vision.pretrained=False
    )


@dataclass
class GenerationHeadCfg:
    """ViT5 seq2seq head (Sec. 6.1)."""

    name: str = "VietAI/vit5-large"
    pretrained: bool = True
    tiny_config: dict[str, Any] | None = (
        None  # T5Config overrides when pretrained=False (must set decoder_start_token_id)
    )
    bridge_norm: bool = False  # LayerNorm after the 768->d_model bridge
    gradient_checkpointing: bool = False


@dataclass
class ClassifierHeadCfg:
    """ViMACSA aspect-sentiment head (Sec. 6.1, Fig. 4c)."""

    num_labels: int = 4  # None, Negative, Neutral, Positive (legacy protocol)
    num_aspects: int = 6
    pool: Literal["mean_joint", "mean_text", "concat_mean_v_t"] = "mean_joint"
    dropout: float = 0.1


@dataclass
class ModelCfg:
    vision: VisionCfg = field(default_factory=VisionCfg)
    text: TextCfg = field(default_factory=TextCfg)
    resampler: ResamplerCfg = field(default_factory=ResamplerCfg)
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    moe: MoECfg | None = None  # None -> dense backbone
    stage1: Stage1Cfg = field(default_factory=Stage1Cfg)
    generation: GenerationHeadCfg = field(default_factory=GenerationHeadCfg)
    classifier: ClassifierHeadCfg = field(default_factory=ClassifierHeadCfg)
    init_std: float = 0.02  # for W_v, W_t, resampler, f_enc, modality embeddings
    logit_scale_init: float = 0.07  # temperature tau (Eq. 13); parameter stores log(1/tau)
    logit_scale_max: float = 100.0  # exp(logit_scale) clamp, in linear units
    modality_embed_in_single_pass: bool = True  # add e_v/e_t in the separate contrastive passes too
    from_checkpoint: str | None = (
        None  # inherit the whole model section from a checkpoint (see loader)
    )

    @property
    def is_moe(self) -> bool:
        return self.moe is not None


# --------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------
@dataclass
class SplitSpec:
    file: str = ""  # relative to DataSourceCfg.root
    image_root: str | None = None  # relative to root; None -> root


@dataclass
class HoldoutSpec:
    """Deterministic dev holdout by image id (Appendix A: 20% of KTVIC train)."""

    fraction: float | None = 0.2
    num_images: int | None = None
    seed: int = 0
    from_split: str = "train"
    to_split: str = "dev"
    ids_file: str | None = None  # authoritative id list (committed under configs/data/splits/)
    exclude_ids_file: str | None = (
        None  # image ids to drop from every split (pretraining-overlap removal)
    )


@dataclass
class DataSourceCfg:
    """One dataset source. ``kind`` selects the adapter; ``options`` are validated by that adapter."""

    kind: str  # registry key: phomt | json_dict | jsonl | hf_image_caption | mixture | ktvic | uitviic | vimacsa | vqa_generic | synthetic_*
    split: str = "train"
    root: str = ""
    splits: dict[str, SplitSpec] = field(default_factory=dict)
    image_check: Literal["none", "exists", "decode"] = "exists"
    missing_image_policy: Literal["skip", "error"] = "error"  # pretraining corpora use skip
    holdout: HoldoutSpec | None = None
    text_normalize: str = "none"  # preset name: none | pretrain | vimacsa_legacy
    manifest_cache: str | None = None
    limit: int | None = None  # keep only the first N records (smoke tests)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenizerCfg:
    vi_max_len: int = 64  # XLM-R captions / questions
    en_max_len: int = 77  # CLIP teacher (hard maximum 77)
    decoder_max_len: int = 64  # ViT5 targets
    cls_max_len: int = 170  # ViMACSA per-aspect sequences (legacy parity)


@dataclass
class TransformCfg:
    # clip_processor: read the CLIP tower's processor; tower: per vision.kind (SigLIP / C-RADIO);
    # manual: explicit size with CLIP statistics
    source: Literal["clip_processor", "tower", "manual"] = "clip_processor"
    size: int | None = None  # manual only
    crop_size: int | None = None
    augment: Literal["none", "random_resized_crop"] = "none"
    draft_decode: bool = True  # PIL JPEG draft downscale-on-decode


@dataclass
class VisionFeatureCacheCfg:
    mode: Literal["disabled", "eval_only", "all"] = (
        "disabled"  # precomputed frozen-ViT tokens (M9, opt-in); never use YAML-boolean words (off/on) as values
    )
    dir: str = "cache/vision_features"


@dataclass
class DataCfg:
    train: DataSourceCfg | None = None
    evals: dict[str, DataSourceCfg] = field(default_factory=dict)
    batch_size: int = 32  # per-rank micro-batch
    eval_batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    one_caption_per_image: bool = (
        True  # contrastive train sampler (one caption per image per epoch)
    )
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    transform: TransformCfg = field(default_factory=TransformCfg)
    vision_features: VisionFeatureCacheCfg = field(default_factory=VisionFeatureCacheCfg)


# --------------------------------------------------------------------------------------------
# optimisation
# --------------------------------------------------------------------------------------------
@dataclass
class ParamGroupCfg:
    modules: list[str]  # dotted module paths (validated with get_submodule)
    lr: float
    weight_decay: float | None = None  # None -> optim.weight_decay
    name: str | None = None


@dataclass
class OptimCfg:
    name: Literal["adamw"] = "adamw"
    lr: float = 1.0e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    weight_decay: float = 0.01
    groups: list[ParamGroupCfg] = field(default_factory=list)  # empty -> single group at optim.lr
    freeze: list[str] = field(default_factory=list)  # dotted module paths frozen for this run
    freeze_preset: str | None = (
        None  # stage1 | stage2 | stage3 | sparse | downstream_seq2seq | downstream_cls
    )
    no_decay_norm_and_bias: bool = True  # legacy-code convention, documented
    strict_group_coverage: bool = (
        True  # every trainable param must belong to a group when groups are given
    )


@dataclass
class ScheduleCfg:
    kind: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_ratio: float | None = 0.1  # paper: 10% linear warmup
    warmup_steps: int | None = None
    min_lr_ratio: float = 0.0


@dataclass
class AmpCfg:
    enabled: bool = True
    dtype: Literal["bf16", "fp16"] = "bf16"  # bf16 on A40; disabled automatically on CPU


@dataclass
class EarlyStoppingCfg:
    enabled: bool = False
    patience: int = 3  # evaluations without improvement
    min_delta: float = 0.0


@dataclass
class TrainCfg:
    max_steps: int | None = None  # exactly one of max_steps / epochs
    epochs: int | None = None
    grad_accum_steps: int = 1
    max_grad_norm: float | None = 1.0  # legacy-code convention, documented
    amp: AmpCfg = field(default_factory=AmpCfg)
    log_every: int = 50
    early_stopping: EarlyStoppingCfg = field(default_factory=EarlyStoppingCfg)
    deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = True
    max_consecutive_nonfinite: int = 5  # fp16 only: skipped micro-batches before aborting


@dataclass
class EvalCfg:
    every_steps: int | None = None
    every_epochs: int | None = 1
    at_start: bool = False
    selection_metric: str | None = None  # "<split>/<metric>", e.g. ktvic_dev/rsum
    selection_mode: Literal["max", "min"] = "max"
    final_splits: list[str] = field(default_factory=list)  # evaluated once with best.pt at the end
    save_predictions: bool = True


@dataclass
class CheckpointCfg:
    every_steps: int | None = None
    every_epochs: int | None = 1
    keep_last_n: int = 2
    save_best: bool = True  # requires eval.selection_metric
    save_optimizer: Literal["always", "last_only", "never"] = "last_only"
    exclude_pristine_frozen_towers: bool = (
        True  # drop frozen towers whose provenance is still the HF checkpoint
    )


@dataclass
class InitFromCfg:
    path: str
    load: list[str] = field(
        default_factory=lambda: ["all"]
    )  # module prefixes to load strictly, or ["all"]


@dataclass
class WandbCfg:
    enabled: bool = False
    project: str = "vivlmoe"
    entity: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class LoggingCfg:
    tensorboard: bool = True
    wandb: WandbCfg = field(default_factory=WandbCfg)
    log_grad_norm: bool = True
    log_memory: bool = True
    log_param_groups: bool = True


@dataclass
class DistCfg:
    backend: Literal["auto", "nccl", "gloo"] = "auto"
    find_unused_parameters: bool = False  # escape hatch only; the model is designed not to need it
    static_graph: bool = False
    broadcast_buffers: bool = (
        False  # router biases are kept identical by all-reduced updates instead
    )
    bucket_cap_mb: int = 25
    timeout_minutes: int = 30
    grad_compression_bf16: bool = False  # optional DDP comm hook for PCIe-only boxes


# --------------------------------------------------------------------------------------------
# losses / analysis / task
# --------------------------------------------------------------------------------------------
@dataclass
class ContrastiveLossCfg:
    gather: bool = True  # grad-flowing all_gather of features across ranks (global-batch negatives)
    symmetric_reduction: Literal["mean", "sum"] = (
        "mean"  # mean = 0.5*(CE_i2t + CE_t2i) (legacy calibration of lambdas)
    )


@dataclass
class RouterLossCfg:
    lambda_z: float = 1.0e-3  # Eq. 20 weight
    lambda_mi: float = 5.0e-3  # Eq. 24 weight
    lambda_lb: float = 0.0  # Switch-style Eq. 19 (ablation arm only)
    use_z: bool = True
    use_mi: bool = True
    use_lb: bool = False
    layer_reduction: Literal["mean", "sum"] = "mean"


@dataclass
class LossCfg:
    contrastive: ContrastiveLossCfg = field(default_factory=ContrastiveLossCfg)
    router: RouterLossCfg = field(default_factory=RouterLossCfg)


@dataclass
class RoutingProbeCfg:
    """Sec. 6.5 routing analysis on a fixed dev token stream."""

    split: str  # name of a data.evals entry
    every_steps: int = 500
    max_steps: int | None = None  # stop probing after this many optimizer steps (paper: first 5K)
    max_batches: int | None = None  # bound the dev subset


@dataclass
class AnalysisCfg:
    routing_probe: RoutingProbeCfg | None = None


@dataclass
class GenerationCfg:
    num_beams: int = 5
    max_new_tokens: int = 40
    min_new_tokens: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = True
    no_repeat_ngram_size: int = 0
    repetition_penalty: float = 1.0


@dataclass
class MetricTokenizerCfg:
    mode: Literal["13a", "vi_safe", "underthesea", "rouge_default"] = "vi_safe"
    lowercase: bool = True
    nfc: bool = True
    strip_punct: bool = True


DEFAULT_ASPECTS = ["Location", "Food", "Room", "Facilities", "Service", "Public_area"]


@dataclass
class TaskCfg:
    """Task section. Fields apply per task as noted; unrelated fields are ignored."""

    name: str  # stage1_teacher | contrastive | captioning | vqa | vimacsa
    metrics: list[str] = field(
        default_factory=list
    )  # generation: bleu4 rougeL cider f1 em ; vimacsa: macro_f1
    max_source_len: int = 64  # vqa question tokens
    max_target_len: int = 48  # captioning / vqa decoder targets
    prompt_template: str = "{question}"  # vqa
    train_on_all_answers: bool = False  # vqa: expand every gold answer into an example
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    metric_tokenizer: MetricTokenizerCfg = field(default_factory=MetricTokenizerCfg)
    bleu_tokenizer: Literal["13a", "vi_safe", "underthesea"] = (
        "13a"  # paper-compat default (HF evaluate)
    )
    report_paper_compat_rouge: bool = (
        True  # also report rougeL with rouge_score's default tokenizer
    )
    aspects: list[str] = field(default_factory=lambda: list(DEFAULT_ASPECTS))  # vimacsa
    use_image_tags: bool = True  # vimacsa: resnet152 image aspect tags as second segment
    max_images: int = 6  # vimacsa
    legacy_parity: bool = False  # vimacsa: pad to cls_max_len like the legacy code
    exclude_none_in_f1: bool = False  # vimacsa macro-F1 over 3 polarities instead of 4 labels
    eval_clip_retrieval: bool = False  # stage1: Appendix-A retrieval in CLIP space on KTVIC dev
    allow_stabilizer_override: bool = (
        False  # permit loss.router / moe.aux_free_bias to differ from the checkpoint
    )


# --------------------------------------------------------------------------------------------
# root
# --------------------------------------------------------------------------------------------
@dataclass
class RunConfig:
    run: RunCfg = field(default_factory=RunCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    task: TaskCfg = field(default_factory=lambda: TaskCfg(name="contrastive"))
    data: DataCfg = field(default_factory=DataCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    checkpoint: CheckpointCfg = field(default_factory=CheckpointCfg)
    init_from: InitFromCfg | None = None
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    dist: DistCfg = field(default_factory=DistCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    analysis: AnalysisCfg = field(default_factory=AnalysisCfg)
    model_overrides: dict[str, Any] = field(
        default_factory=dict
    )  # applied over model.from_checkpoint (loader)

    def validate(self) -> None:
        """Cross-field rules. Raises :class:`ConfigError` with the offending key path."""
        t = self.train
        if (t.max_steps is None) == (t.epochs is None):
            raise ConfigError("train: set exactly one of max_steps / epochs")
        if t.max_steps is not None and t.max_steps < 1:
            raise ConfigError("train.max_steps must be >= 1")
        if t.epochs is not None and t.epochs < 1:
            raise ConfigError("train.epochs must be >= 1")
        if t.grad_accum_steps < 1:
            raise ConfigError("train.grad_accum_steps must be >= 1")
        s = self.schedule
        if (s.warmup_ratio is None) == (s.warmup_steps is None):
            raise ConfigError("schedule: set exactly one of warmup_ratio / warmup_steps")
        if s.warmup_ratio is not None and not 0.0 <= s.warmup_ratio < 1.0:
            raise ConfigError("schedule.warmup_ratio must be in [0, 1)")
        if t.early_stopping.enabled and self.eval.selection_metric is None:
            raise ConfigError("train.early_stopping.enabled requires eval.selection_metric")
        if self.eval.selection_metric is not None and "/" not in self.eval.selection_metric:
            raise ConfigError("eval.selection_metric must be '<split>/<metric>'")
        if self.eval.selection_metric is not None:
            split = self.eval.selection_metric.split("/", 1)[0]
            if split not in self.data.evals:
                raise ConfigError(
                    f"eval.selection_metric refers to split '{split}' which is not in data.evals {sorted(self.data.evals)}"
                )
        for split in self.eval.final_splits:
            if split not in self.data.evals:
                raise ConfigError(
                    f"eval.final_splits: '{split}' is not in data.evals {sorted(self.data.evals)}"
                )
        if t.amp.enabled and t.amp.dtype == "fp16" and self.model.vision.frozen_dtype == "bf16":
            raise ConfigError(
                "model.vision.frozen_dtype=bf16 cannot be combined with train.amp.dtype=fp16"
            )
        v = self.model.vision
        if v.kind != "clip" and v.apply_post_layernorm:
            raise ConfigError(
                "model.vision.apply_post_layernorm only applies to CLIP towers (SigLIP's last hidden "
                "state is already post-LN; C-RADIO returns normalised features)"
            )
        if v.kind == "radio" and v.gradient_checkpointing:
            raise ConfigError(
                "model.vision: C-RADIO towers are frozen-only (no gradient checkpointing)"
            )
        if v.kind != "clip" and self.data.transform.source == "clip_processor":
            raise ConfigError(
                f"data.transform.source=clip_processor cannot describe a {v.kind} vision tower "
                "(it would resize and centre-crop to the wrong size): use source: tower"
            )
        if self.data.batch_size < 1 or self.data.eval_batch_size < 1:
            raise ConfigError("data.batch_size / data.eval_batch_size must be >= 1")
        if self.data.num_workers < 0:
            raise ConfigError("data.num_workers must be >= 0")
        if self.init_from is not None and not self.init_from.load:
            raise ConfigError("init_from.load must list module prefixes or ['all']")
        if self.checkpoint.keep_last_n < 1:
            raise ConfigError("checkpoint.keep_last_n must be >= 1")
        if self.analysis.routing_probe is not None:
            rp = self.analysis.routing_probe
            if rp.split not in self.data.evals:
                raise ConfigError(f"analysis.routing_probe.split '{rp.split}' is not in data.evals")
            if rp.every_steps < 1:
                raise ConfigError("analysis.routing_probe.every_steps must be >= 1")
        if self.model.from_checkpoint is None and self.model_overrides:
            raise ConfigError("model_overrides requires model.from_checkpoint")
        if self.task.name == "vimacsa" and self.model.classifier.num_aspects != len(
            self.task.aspects
        ):
            raise ConfigError("model.classifier.num_aspects must equal len(task.aspects)")
