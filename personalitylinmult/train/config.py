"""Typed configuration for the FI training pipeline.

Configuration is plain YAML parsed into frozen dataclasses. Two properties
motivate that choice over a framework:

* The same :class:`~personalitylinmult.data.schema.FeatureSpec` objects describe
  a feature to the HDF5 builder and to the dataloader. A framework that derives
  its schema from class constructors (Lightning CLI / jsonargparse) could not
  span both, because the builder is not a Lightning class.
* Nothing here changes the working directory or writes hidden output trees, so
  MLflow artifact paths stay predictable.

Every knob a run depends on lives in one of these dataclasses, and
:meth:`ExperimentConfig.to_flat_dict` flattens the lot into MLflow params.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from personalitylinmult.data.schema import (
    ALL_TRAITS,
    BIG_FIVE_TRAITS,
    CROP_MODES,
    INTERVIEW_TRAITS,
    FeatureSpec,
)

logger = logging.getLogger(__name__)
"""Module-level logger."""

TASKS: frozenset[str] = frozenset({"bigfive", "interview"})
"""The two regression tasks.

They are trained separately and never share a head: big-five perception and the
interview (hireability) judgement are different prediction problems that happen
to share a feature pipeline.
"""

TASK_TARGETS: dict[str, tuple[str, ...]] = {
    "bigfive": BIG_FIVE_TRAITS,
    "interview": INTERVIEW_TRAITS,
}
"""Default targets per task."""

OPTIMIZERS: frozenset[str] = frozenset({"adam", "adamw", "radam"})
"""Supported optimizers."""

SCHEDULERS: frozenset[str] = frozenset({"onecycle", "cosine", "plateau", "none"})
"""Supported LR schedulers."""

LOSSES: frozenset[str] = frozenset({"bell_l2_l1", "l1", "mse", "huber", "ccc", "ccc+l1"})
"""Supported losses."""

PRECISIONS: frozenset[str] = frozenset(
    {"32-true", "16-mixed", "bf16-mixed", "64-true", "16-true", "bf16-true"}
)
"""Lightning precision settings this project supports.

Validated here so a typo fails at config load with a readable message, rather
than deep inside the Trainer.
"""


MODEL_FAMILIES = frozenset({"linmult", "lint"})
"""Architecture families a model config may select.

``linmult`` builds the multimodal cross-modal transformer; ``lint`` builds the
single-stream transformer used for the unimodal ablation arms. The two differ
in how many features they accept and in their forward signatures, so the family
is resolved once at config time rather than inferred later.
"""


class ConfigError(ValueError):
    """Raised when a configuration is invalid or self-contradictory."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Parse a YAML file into a dict.

    Args:
        path (str | Path): File to read.

    Returns:
        dict: Parsed mapping.

    Raises:
        ConfigError: If the file is missing or is not a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    parsed = yaml.safe_load(path.read_text())
    if not isinstance(parsed, dict):
        raise ConfigError(f"{path}: expected a YAML mapping, got {type(parsed).__name__}.")
    return parsed


def _reject_unknown(cls: type, data: dict[str, Any], context: str) -> None:
    """Fail on config keys the dataclass does not declare.

    A silently-ignored key is a config that lies about what a run did, which is
    worse than a crash.

    Args:
        cls (type): Target dataclass.
        data (dict): Incoming values.
        context (str): Human-readable source, for the error message.

    Raises:
        ConfigError: If any key is unknown.
    """
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"{context}: unknown keys {unknown}. Known keys: {sorted(known)}.")


@dataclass(frozen=True)
class DataConfig:
    """Dataloader configuration.

    Args:
        h5_path (str): The built dataset; the only file training reads.
        features (list[str]): Features to feed the model, in input order.
        window_seconds (float | None): Analysis window. Changing this reruns the
            window experiment without rebuilding the HDF5.
        crop_mode (str): ``head``, ``center``, or ``random``.
        batch_size (int): Clips per batch.
        num_workers (int): Dataloader worker processes.
        pin_memory (bool): Page-lock host batches for faster device copies.
        persistent_workers (bool): Keep workers (and their HDF5 handles) alive
            between epochs.
        prefetch_factor (int): Batches prefetched per worker.
        seed (int): Seed for shuffling and random cropping.
    """

    h5_path: str = "data/processed/FI/fi.h5"
    features: list[str] = field(default_factory=list)
    window_seconds: float | None = 15.0
    crop_mode: str = "head"
    batch_size: int = 16
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate the dataloader configuration.

        Raises:
            ConfigError: If any value is out of range or unrecognised.
        """
        if not self.features:
            raise ConfigError("data.features must name at least one feature.")
        duplicates = sorted({f for f in self.features if self.features.count(f) > 1})
        if duplicates:
            raise ConfigError(f"data.features contains duplicates: {duplicates}.")
        if self.crop_mode not in CROP_MODES:
            raise ConfigError(
                f"data.crop_mode={self.crop_mode!r} is not one of {sorted(CROP_MODES)}."
            )
        if self.batch_size < 1:
            raise ConfigError(f"data.batch_size must be >= 1, got {self.batch_size}.")
        if self.num_workers < 0:
            raise ConfigError(f"data.num_workers must be >= 0, got {self.num_workers}.")
        if self.window_seconds is not None and self.window_seconds <= 0:
            raise ConfigError(
                f"data.window_seconds must be positive or null, got {self.window_seconds}."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataConfig:
        """Build from a plain dict.

        Args:
            data (dict): Config values.

        Returns:
            DataConfig: The validated config.
        """
        _reject_unknown(cls, data, "data config")
        return cls(**data)


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer and LR-schedule configuration.

    Args:
        name (str): Optimizer name.
        lr (float): Peak learning rate.
        weight_decay (float): L2 / decoupled weight decay.
        scheduler (str): Schedule name, or ``"none"``.
        warmup_ratio (float): Fraction of training spent warming up, for
            schedules that support it.
        gradient_clip_val (float): Gradient-norm clip; ``0`` disables.
    """

    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.01
    scheduler: str = "onecycle"
    warmup_ratio: float = 0.1
    gradient_clip_val: float = 1.0

    def __post_init__(self) -> None:
        """Validate the optimizer configuration.

        Raises:
            ConfigError: If any value is out of range or unrecognised.
        """
        if self.name not in OPTIMIZERS:
            raise ConfigError(f"optimizer.name={self.name!r} not in {sorted(OPTIMIZERS)}.")
        if self.scheduler not in SCHEDULERS:
            raise ConfigError(
                f"optimizer.scheduler={self.scheduler!r} not in {sorted(SCHEDULERS)}."
            )
        if self.lr <= 0:
            raise ConfigError(f"optimizer.lr must be positive, got {self.lr}.")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ConfigError(f"optimizer.warmup_ratio must be in [0, 1), got {self.warmup_ratio}.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimConfig:
        """Build from a plain dict.

        Args:
            data (dict): Config values.

        Returns:
            OptimConfig: The validated config.
        """
        _reject_unknown(cls, data, "optimizer config")
        return cls(**data)


@dataclass(frozen=True)
class ModelConfig:
    """LinMulT architecture configuration.

    Field names track **linmult 2.x** exactly. The 1.5.2-era names used by the
    old configs (``n_heads``, ``n_layers``, ``dropout_qkv``, ``output_dim``,
    ``ffn_fusion``) no longer exist upstream and are rejected rather than
    silently ignored.

    ``input_feature_dim`` is absent by design: it is derived from the HDF5's
    feature attributes so the model can never disagree with its data.

    Args:
        name (str): Model name, recorded in the run.
        family (str): ``linmult`` (multimodal) or ``lint`` (single-stream).
            Selects the model class; LinT accepts exactly one feature.
        d_model (int): Internal width.
        num_heads (int): Attention heads.
        cmt_num_layers (int): Cross-modal (LinMulT) / encoder (LinT) depth.
        branch_sat_num_layers (int): Per-branch self-attention depth. LinMulT
            only; ignored by LinT.
        attention_type (str): ``linear`` (LinMulT) or ``softmax`` (MulT); this
            is the only architectural difference between the two baselines.
            ``flash``, ``performer``, and ``bigbird`` are also accepted, each
            reading its matching knobs below.
        flash_query_key_dim (int | None): Flash-attention query/key width.
        performer_num_random_features (int | None): Performer feature count.
        bigbird_block_size (int): BigBird block size.
        bigbird_num_global_tokens (int): BigBird global-token count.
        bigbird_num_random_tokens (int): BigBird random-token count.
        dropout_input (float): Dropout before projection.
        dropout_output (float): Dropout after fusion.
        dropout_pe (float): Dropout after positional encoding.
        dropout_ffn (float): Dropout inside FFN blocks.
        dropout_attention (float): Attention-weight dropout.
        add_module_tcn (bool): Per-modality TCN after projection. linmult 2.x
            defaults this to ``True``; the published 1.5.2 results had no TCN,
            so this config defaults it to ``False`` and makes enabling it a
            deliberate experiment.
        tcn_num_layers (int): TCN depth, when the TCN is enabled.
        tcn_kernel_size (int): TCN kernel size.
        tcn_dropout (float): TCN dropout.
        add_module_multimodal_signal (bool): Cross-modal MM-signal module. Off by
            default so the baseline is the plain cross-modal transformer; the
            ablation flips it to measure its contribution. LinMulT-only.
        mms_num_layers (int): MM-signal module depth, when it is enabled.
        time_dim_reducer (str): Collapses ``(B, T, F)`` to ``(B, F)`` before the
            head. ``gap`` matches the original configuration.
        add_module_ffn_fusion (bool): FFN + residual after fusion.
        head_type (str): linmult head type.
        head_dropout (float): Dropout inside the head.
        head_hidden_dim (int): Head hidden width.
        head_norm (str): Head normalisation (``bn`` batch-norm, ``ln`` layer-norm).
    """

    name: str = "LinMulT"
    # `family` selects the architecture: 'linmult' for the multimodal
    # cross-modal transformer, 'lint' for the single-stream transformer used in
    # the unimodal ablation arms. LinT takes exactly one feature; LinMulT takes
    # one or more. This is the only field that changes the model class built.
    family: str = "linmult"
    d_model: int = 32
    num_heads: int = 8
    cmt_num_layers: int = 5
    # LinMulT-only: depth of the per-branch self-attention transformer. Ignored
    # by LinT, which has no cross-modal branches.
    branch_sat_num_layers: int = 5
    attention_type: str = "linear"
    # Attention-variant knobs, passed through to whichever family is built.
    # Only consulted when the matching attention_type is selected.
    flash_query_key_dim: int | None = None
    performer_num_random_features: int | None = None
    bigbird_block_size: int = 64
    bigbird_num_global_tokens: int = 16
    bigbird_num_random_tokens: int = 10
    dropout_input: float = 0.0
    dropout_output: float = 0.2
    dropout_pe: float = 0.1
    dropout_ffn: float = 0.2
    dropout_attention: float = 0.0
    # Per-modality TCN after projection. linmult 2.x defaults this True; the
    # published 1.5.2 results had no TCN, so parity here is the explicit False.
    add_module_tcn: bool = False
    tcn_num_layers: int = 3
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.1
    # Cross-modal multimodal-signal module. Off by default so the baseline is the
    # plain cross-modal transformer; the ablation flips this to measure whether
    # the MM-signal module earns its place, the same way add_module_tcn does.
    add_module_multimodal_signal: bool = False
    mms_num_layers: int = 6
    time_dim_reducer: str = "gap"
    add_module_ffn_fusion: bool = True
    head_type: str = "simple"
    head_dropout: float = 0.2
    head_hidden_dim: int = 256
    head_norm: str = "bn"

    def __post_init__(self) -> None:
        """Validate the architecture configuration.

        Raises:
            ConfigError: If a value is out of range or the family is unknown.
        """
        if self.family not in MODEL_FAMILIES:
            raise ConfigError(
                f"model.family={self.family!r} is unknown; expected one of "
                f"{sorted(MODEL_FAMILIES)}."
            )
        if self.d_model < 1:
            raise ConfigError(f"model.d_model must be >= 1, got {self.d_model}.")
        if self.d_model % self.num_heads:
            raise ConfigError(
                f"model.d_model={self.d_model} must be divisible by num_heads={self.num_heads}."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Build from a plain dict.

        Args:
            data (dict): Config values.

        Returns:
            ModelConfig: The validated config.

        Raises:
            ConfigError: If a retired linmult 1.x key is present.
        """
        legacy = {
            "n_heads": "num_heads",
            "n_layers": "cmt_num_layers",
            "ffn_fusion": "add_module_ffn_fusion",
            "output_dim": "(derived from the task's targets)",
            "dropout_qkv": "dropout_attention",
            "dropout_relu": "dropout_ffn",
            "dropout_residual": "dropout_ffn",
            "version": "(no longer used)",
            "input_feature_dim": "(derived from the h5 feature attrs)",
        }
        found = sorted(set(data) & set(legacy))
        if found:
            renames = ", ".join(f"{k!r} -> {legacy[k]}" for k in found)
            raise ConfigError(
                f"model config uses linmult 1.x keys that no longer exist in "
                f"linmult 2.x: {renames}. Rewrite the config against the 2.x API."
            )
        _reject_unknown(cls, data, "model config")
        return cls(**data)


@dataclass(frozen=True)
class MLflowConfig:
    """MLflow tracking configuration.

    Args:
        tracking_uri (str): Backend store. A **SQLite database, not a directory**:
            MLflow 3.x puts the filesystem store (``./mlruns``) in maintenance
            mode and raises rather than using it. Relative by default so no
            absolute local path leaks into a committed config.
        experiment_name (str): MLflow experiment to log under.
        run_name (str | None): Run name; MLflow generates one when ``None``.
        log_model (bool): Log the best checkpoint as an MLflow model.
    """

    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "FI"
    run_name: str | None = None
    log_model: bool = True

    def __post_init__(self) -> None:
        """Warn when the retired filesystem store is configured.

        Raises:
            ConfigError: If the tracking URI is a bare directory path, which
                MLflow 3.x refuses to use.
        """
        uri = self.tracking_uri
        if "://" not in uri:
            raise ConfigError(
                f"mlflow.tracking_uri={uri!r} looks like a filesystem store. "
                "MLflow 3.x has retired the file store and raises on it; use a "
                "database URI such as 'sqlite:///mlflow.db'."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLflowConfig:
        """Build from a plain dict.

        Args:
            data (dict): Config values.

        Returns:
            MLflowConfig: The validated config.
        """
        _reject_unknown(cls, data, "mlflow config")
        return cls(**data)


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early-stopping configuration.

    Args:
        enabled (bool): Whether to stop early at all.
        monitor (str): Metric to watch.
        mode (str): ``"max"`` or ``"min"``.
        patience (int): Epochs without improvement before stopping.
        min_delta (float): Smallest change counted as an improvement.
    """

    enabled: bool = True
    monitor: str = "valid/mean_ccc"
    mode: str = "max"
    patience: int = 10
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        """Validate the early-stopping configuration.

        Raises:
            ConfigError: If ``mode`` is not ``"min"`` or ``"max"``.
        """
        if self.mode not in {"min", "max"}:
            raise ConfigError(f"early_stopping.mode must be 'min' or 'max', got {self.mode!r}.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EarlyStoppingConfig:
        """Build from a plain dict.

        Args:
            data (dict): Config values.

        Returns:
            EarlyStoppingConfig: The validated config.
        """
        _reject_unknown(cls, data, "early_stopping config")
        return cls(**data)


@dataclass(frozen=True)
class TrainConfig:
    """Training-loop configuration.

    Args:
        task (str): ``"bigfive"`` or ``"interview"``.
        targets (list[str]): Label columns to regress. Defaults to the task's
            standard targets.
        loss (str): Loss name.
        max_epochs (int): Training epochs.
        accelerator (str): Lightning accelerator.
        devices (list[int] | int | str): Lightning devices spec.
        precision (str): Lightning precision.
        seed (int): Global seed.
        deterministic (bool): Request deterministic kernels.
        output_dir (str): Root for run outputs.
        accumulate_grad_batches (int): Optimizer steps span this many forward
            passes before a weight update, giving an effective batch size of
            ``data.batch_size * accumulate_grad_batches`` without holding more
            than one physical batch in memory at once. Losses computed *per
            batch* (e.g. ``ccc``, whose mean/var/covariance are batch
            statistics) still only see one physical batch's worth of
            statistics per forward pass -- accumulation stabilises the
            *gradient*, not the per-step CCC estimate itself. A true larger
            physical batch (raising ``data.batch_size``) is the only way to
            reduce CCC's own statistical noise; accumulation is what to use
            once the physical batch size that would require is too large to
            fit in memory.
        optimizer (OptimConfig): Optimizer settings.
        early_stopping (EarlyStoppingConfig): Early-stopping settings.
        mlflow (MLflowConfig): Tracking settings.
    """

    task: str = "bigfive"
    targets: list[str] = field(default_factory=list)
    loss: str = "bell_l2_l1"
    max_epochs: int = 30
    accelerator: str = "auto"
    devices: list[int] | int | str = "auto"
    precision: str = "32-true"
    seed: int = 42
    deterministic: bool = False
    output_dir: str = "results/FI"
    accumulate_grad_batches: int = 1
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    def __post_init__(self) -> None:
        """Validate the training configuration.

        Raises:
            ConfigError: If the task, targets, or loss are invalid.
        """
        if self.task not in TASKS:
            raise ConfigError(f"train.task={self.task!r} not in {sorted(TASKS)}.")
        if self.loss not in LOSSES:
            raise ConfigError(f"train.loss={self.loss!r} not in {sorted(LOSSES)}.")
        if self.max_epochs < 1:
            raise ConfigError(f"train.max_epochs must be >= 1, got {self.max_epochs}.")
        if self.accumulate_grad_batches < 1:
            raise ConfigError(
                f"train.accumulate_grad_batches must be >= 1, got {self.accumulate_grad_batches}."
            )
        if self.precision not in PRECISIONS:
            raise ConfigError(f"train.precision={self.precision!r} not in {sorted(PRECISIONS)}.")

        if not self.targets:
            object.__setattr__(self, "targets", list(TASK_TARGETS[self.task]))

        unknown = sorted(set(self.targets) - set(ALL_TRAITS))
        if unknown:
            raise ConfigError(
                f"train.targets contains unknown labels {unknown}. Available: {sorted(ALL_TRAITS)}."
            )

        if self.task == "bigfive" and "neuroticism" in self.targets:
            raise ConfigError(
                "train.targets uses 'neuroticism', but the big-five task trains on "
                "'emotional_stability' (= 1 - neuroticism) so that every trait points "
                "the same way. Use 'emotional_stability'."
            )

        if self.task == "interview" and set(self.targets) != set(INTERVIEW_TRAITS):
            raise ConfigError(
                f"train.task='interview' regresses {list(INTERVIEW_TRAITS)}, but "
                f"targets={self.targets}. The interview variable is a separate task "
                "from big-five regression and is trained on its own."
            )

    @property
    def output_dim(self) -> int:
        """Head width, derived from the task's targets.

        Returns:
            int: Number of regressed values.
        """
        return len(self.targets)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        """Build from a plain dict, including nested sections.

        Args:
            data (dict): Config values.

        Returns:
            TrainConfig: The validated config.
        """
        data = dict(data)
        _reject_unknown(cls, data, "train config")
        if "optimizer" in data:
            data["optimizer"] = OptimConfig.from_dict(data["optimizer"])
        if "early_stopping" in data:
            data["early_stopping"] = EarlyStoppingConfig.from_dict(data["early_stopping"])
        if "mlflow" in data:
            data["mlflow"] = MLflowConfig.from_dict(data["mlflow"])
        return cls(**data)


@dataclass(frozen=True)
class ExperimentConfig:
    """One run: its data, its model, and its training procedure.

    Args:
        data (DataConfig): Dataloader settings.
        model (ModelConfig): Architecture settings.
        train (TrainConfig): Training settings.
    """

    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    def __post_init__(self) -> None:
        """Validate consistency across the three sections.

        Raises:
            ConfigError: If the model family and the feature count disagree.
        """
        if self.model.family == "lint" and len(self.data.features) != 1:
            raise ConfigError(
                f"model.family='lint' is single-stream but the data config lists "
                f"{len(self.data.features)} features ({self.data.features}). Use "
                "family='linmult' for more than one feature, or reduce the feature "
                "list to one."
            )

    @classmethod
    def from_files(
        cls,
        data_path: str | Path,
        model_path: str | Path,
        train_path: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> ExperimentConfig:
        """Load a run's three config files, applying dotted overrides.

        Args:
            data_path (str | Path): Data config.
            model_path (str | Path): Model config.
            train_path (str | Path): Train config.
            overrides (dict | None): Dotted ``section.key`` overrides, e.g.
                ``{"data.window_seconds": 10.0}``.

        Returns:
            ExperimentConfig: The assembled configuration.
        """
        raw = {
            "data": load_yaml(data_path),
            "model": load_yaml(model_path),
            "train": load_yaml(train_path),
        }
        # The data config also drives the h5 builder, which owns these keys.
        for key in ("label_csv",):
            raw["data"].pop(key, None)

        for dotted, value in (overrides or {}).items():
            apply_override(raw, dotted, value)

        return cls(
            data=DataConfig.from_dict(raw["data"]),
            model=ModelConfig.from_dict(raw["model"]),
            train=TrainConfig.from_dict(raw["train"]),
        )

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten every field to dotted keys, for ``mlflow.log_params``.

        Returns:
            dict: e.g. ``{"data.batch_size": 16, "train.optimizer.lr": 0.001}``.
        """
        return _flatten(asdict(self))

    def resolve_feature_specs(self, feature_dir: Path) -> list[FeatureSpec]:
        """Load the specs of the features this run consumes, in input order.

        Args:
            feature_dir (Path): Directory of per-feature YAML files.

        Returns:
            list[FeatureSpec]: One spec per configured feature.

        Raises:
            ConfigError: If a feature config is missing.
        """
        specs = []
        for name in self.data.features:
            path = feature_dir / f"{name}.yaml"
            if not path.is_file():
                raise ConfigError(f"No feature config for {name!r} at {path}.")
            specs.append(FeatureSpec.from_dict(load_yaml(path)))
        return specs


def apply_override(raw: dict[str, Any], dotted: str, value: Any) -> None:
    """Set a nested config value addressed by a dotted key.

    Args:
        raw (dict): Nested config mapping, modified in place.
        dotted (str): Key path, e.g. ``"train.optimizer.lr"``.
        value (Any): Value to set.

    Raises:
        ConfigError: If the path does not address a mapping.
    """
    parts = dotted.split(".")
    node = raw
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse a ``key=value`` CLI override, typing the value via YAML.

    Args:
        text (str): e.g. ``"data.window_seconds=10"``.

    Returns:
        tuple[str, Any]: The dotted key and its parsed value.

    Raises:
        ConfigError: If the text has no ``=``.
    """
    if "=" not in text:
        raise ConfigError(f"Override {text!r} must be of the form key=value.")
    key, _, raw = text.partition("=")
    value = yaml.safe_load(raw)

    # PyYAML implements YAML 1.1, whose float pattern requires a decimal point
    # in the mantissa: "1e-4" parses as the string "1e-4", not 0.0001. A learning
    # rate silently arriving as a string is exactly the kind of bug that only
    # surfaces deep inside the optimizer, so recover the intent here.
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass

    return key.strip(), value


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested mapping to dotted keys.

    Args:
        data (dict): Nested mapping.
        prefix (str): Accumulated key prefix.

    Returns:
        dict: Flattened mapping.
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        elif isinstance(value, list):
            flat[path] = ",".join(str(v) for v in value)
        else:
            flat[path] = value
    return flat
