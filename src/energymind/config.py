"""Typed configuration loading and validation.

All pipeline stages are driven by the YAML files in ``configs/``. This module
turns them into validated pydantic models so that a typo or a missing key fails
loudly at startup rather than halfway through a run.

Run it directly to validate every config file::

    python -m energymind.config
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: Repository root (``<root>/src/energymind/config.py`` -> parents[2]).
REPO_ROOT = Path(__file__).resolve().parents[2]
#: Default directory holding the YAML configs.
DEFAULT_CONFIGS_DIR = REPO_ROOT / "configs"

DurationStr = Annotated[str, Field(pattern=r"^\d+(min|h|d)$")]
Origin = Literal["data_quality", "adversarial", "physical_event"]
Variant = Literal["naive", "consistent"]
Risk = Literal["low", "medium", "high"]
Backend = Literal["ollama", "openai_compatible"]


class _Base(BaseModel):
    """Base model: reject unknown keys so config typos surface immediately."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# data.yaml
# --------------------------------------------------------------------------- #


class PathsConfig(_Base):
    raw_csv: str
    manifest: str
    clean_dir: str
    features_dir: str
    injected_dir: str
    splits_file: str


class TimeConfig(_Base):
    timestamp_column: str = "timestamp"
    timezone_offset_column: str = "timezone_offset_h"
    #: The file carries a constant offset and no DST.
    assume_utc_offset_h: int = -6
    freq: str = "30min"
    expected_rows_per_household: int = 17_520


class ColumnsConfig(_Base):
    #: Everything from the start of the file *before* this column is measured or
    #: metadata; everything from here on is a derived feature that must be dropped.
    derived_start: str = "hour"
    drop_also: list[str] = Field(default_factory=list)
    label_columns: list[str] = Field(default_factory=list)
    add_back: list[str] = Field(default_factory=list)


class BoundsConfig(_Base):
    voltage_v: tuple[float, float | None]
    frequency_hz: tuple[float, float]
    power_factor: tuple[float, float]
    active_power_kw: tuple[float, float | None]
    sub_meter_tolerance: float


class DateRange(_Base):
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> DateRange:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be after start ({self.start})")
        return self


class TemporalSplits(_Base):
    train: DateRange
    val: DateRange
    test: DateRange


class SplitsConfig(_Base):
    temporal: TemporalSplits
    holdout_households: list[str] = Field(default_factory=list)


class DataConfig(_Base):
    paths: PathsConfig
    time: TimeConfig
    columns: ColumnsConfig
    bounds: BoundsConfig
    physics_residual_tolerance: float
    splits: SplitsConfig


# --------------------------------------------------------------------------- #
# injection.yaml
# --------------------------------------------------------------------------- #


class DurationRange(_Base):
    min: DurationStr
    max: DurationStr


class InjectionType(_Base):
    origin: Origin
    length: DurationRange | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    variants: list[Variant] = Field(default_factory=lambda: ["naive", "consistent"])
    note: str | None = None


class InjectionConfig(_Base):
    seed_base: int
    variants: list[Variant]
    prevalence: list[float]
    one_anomaly_per_household_interval: bool = True
    inject_only_quality_mask: bool = True
    #: Types hidden from train/val so the "unknown, escalate" path can be tested.
    withhold_from_train: list[str] = Field(default_factory=list)
    types: dict[str, InjectionType]

    @field_validator("prevalence")
    @classmethod
    def _each_prevalence_in_unit_interval(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("prevalence must not be empty")
        for p in v:
            if not 0.0 < p < 1.0:
                raise ValueError(f"prevalence {p} must be in (0, 1)")
        return v

    @model_validator(mode="after")
    def _withheld_types_exist(self) -> InjectionConfig:
        unknown = set(self.withhold_from_train) - set(self.types)
        if unknown:
            raise ValueError(f"withhold_from_train references unknown types: {sorted(unknown)}")
        return self


# --------------------------------------------------------------------------- #
# models.yaml
# --------------------------------------------------------------------------- #


class DetectorSpec(_Base):
    enabled: bool = False
    params: dict[str, Any] = Field(default_factory=dict)


class ClassifierSpec(_Base):
    model: str
    calibration: str = "isotonic"
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(_Base):
    seeds: list[int]
    prevalences: list[float]
    tolerance_intervals: int = 1
    false_alarm_budget_per_household_week: float
    no_point_adjust: bool = True
    metrics: list[str]


class ModelsConfig(_Base):
    detectors: dict[str, DetectorSpec]
    classifier: ClassifierSpec
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def _at_least_one_detector(self) -> ModelsConfig:
        if not any(d.enabled for d in self.detectors.values()):
            raise ValueError("at least one detector must be enabled")
        return self


# --------------------------------------------------------------------------- #
# policy.yaml
# --------------------------------------------------------------------------- #


class ActionSpec(_Base):
    risk: Risk
    reversible: bool


class TiersConfig(_Base):
    #: Confidence above which a low-risk, data_quality, converged case may be autonomous.
    tau_auto: float
    #: Confidence above which a case may go for human approval.
    tau_review: float

    @model_validator(mode="after")
    def _ordered(self) -> TiersConfig:
        if self.tau_auto < self.tau_review:
            raise ValueError("tau_auto must be >= tau_review")
        return self


class PolicyConfig(_Base):
    #: The deliberation round limit lives in AgentsConfig (configs/agents.yaml).
    error_target: float = Field(gt=0.0, lt=1.0)
    tiers: TiersConfig
    actions: dict[str, ActionSpec]
    never_automate: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _never_automate_not_an_action(self) -> PolicyConfig:
        overlap = set(self.never_automate) & set(self.actions)
        if overlap:
            raise ValueError(f"never_automate overlaps the action set: {sorted(overlap)}")
        return self


# --------------------------------------------------------------------------- #
# agents.yaml
# --------------------------------------------------------------------------- #


class AgentSpec(_Base):
    family: str
    model: str


class OpenAICompatibleSpec(_Base):
    base_url: str
    api_key_env: str


class AgentsConfig(_Base):
    backend: Backend
    r_max: int = Field(ge=1)
    timeout_s: float = Field(gt=0)
    temperature: float = Field(ge=0.0, le=2.0)
    seed: int
    require_new_evidence_to_revise: bool = True
    fallback_to_rules: bool = True
    analyst: AgentSpec
    validator: AgentSpec
    openai_compatible: OpenAICompatibleSpec

    @model_validator(mode="after")
    def _distinct_families(self) -> AgentsConfig:
        # Two copies of the same model make correlated errors; the design relies
        # on the agents using different evidence *and* different model families.
        if self.analyst.family == self.validator.family:
            raise ValueError("analyst and validator must use different model families")
        return self


# --------------------------------------------------------------------------- #
# store.yaml
# --------------------------------------------------------------------------- #


class QdrantConfig(_Base):
    #: Cluster endpoint; empty means "use QDRANT_URL from the environment".
    url: str = ""
    api_key_env: str = "QDRANT_API_KEY"
    collection: str = "anomaly_case"
    vector_size: int = 64
    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    local_path: str = ""  # set to run an embedded/local Qdrant (e.g. data/memory/qdrant)


class AuditConfig(_Base):
    backend: Literal["sqlite", "postgres"] = "sqlite"
    sqlite_path: str = "data/memory/audit.db"
    dsn_env: str = "PG_DSN"
    dsn_default: str = "postgresql://energymind:energymind@localhost:5432/energymind"


class RetrievalConfig(_Base):
    k: int = 5
    #: Minimum cosine similarity for a "seen before" match.
    tau_sim: float = 0.9


class StoreConfig(_Base):
    qdrant: QdrantConfig
    audit: AuditConfig
    retrieval: RetrievalConfig


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Settings:
    """Every validated config for a run, in one object."""

    data: DataConfig
    injection: InjectionConfig
    models: ModelsConfig
    policy: PolicyConfig
    agents: AgentsConfig
    store: StoreConfig


_CONFIG_FILES = {
    "data": ("data.yaml", DataConfig),
    "injection": ("injection.yaml", InjectionConfig),
    "models": ("models.yaml", ModelsConfig),
    "policy": ("policy.yaml", PolicyConfig),
    "agents": ("agents.yaml", AgentsConfig),
    "store": ("store.yaml", StoreConfig),
}


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a plain dict, with a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
    return loaded


def load_settings(configs_dir: Path | str | None = None) -> Settings:
    """Load and validate every config file in ``configs_dir``."""
    directory = Path(configs_dir) if configs_dir is not None else DEFAULT_CONFIGS_DIR
    loaded: dict[str, Any] = {}
    for key, (filename, model) in _CONFIG_FILES.items():
        raw = load_yaml(directory / filename)
        loaded[key] = model.model_validate(raw)
    return Settings(**loaded)


def _summarize(settings: Settings) -> str:
    inj = settings.injection
    models = settings.models
    enabled = [name for name, spec in models.detectors.items() if spec.enabled]
    return "\n".join(
        [
            f"config dir      : {DEFAULT_CONFIGS_DIR}",
            f"raw csv         : {settings.data.paths.raw_csv}",
            f"injection types : {len(inj.types)} "
            f"({', '.join(sorted({t.origin for t in inj.types.values()}))})",
            f"prevalence      : {inj.prevalence}",
            f"detectors on    : {', '.join(enabled)}",
            f"classifier      : {models.classifier.model}",
            f"tiers           : tau_auto={settings.policy.tiers.tau_auto} "
            f"tau_review={settings.policy.tiers.tau_review}",
            f"agents          : {settings.agents.analyst.model} (A) vs "
            f"{settings.agents.validator.model} (B) via {settings.agents.backend}",
        ]
    )


def main() -> int:
    """Validate every config file and print a short summary."""
    settings = load_settings()
    print("Configuration OK")
    print(_summarize(settings))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
