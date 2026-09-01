"""Application configuration.

Loads:

* Environment variables (via :mod:`pydantic_settings`).
* Static YAML configuration under :data:`CONFIG_DIR` (``configs/``).

Phase 0 implements the loading machinery only — the actual sections
(``assets``, ``features``, ``models``) are wired up by later phases.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This file lives at: <repo>/src/financial_ml/config.py
# Repo root is three parents up from here.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
SRC_DIR: Path = Path(__file__).resolve().parents[1]
PACKAGE_DIR: Path = Path(__file__).resolve().parent

CONFIG_DIR: Path = REPO_ROOT / "configs"
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
FEATURE_DATA_DIR: Path = DATA_DIR / "features"
MODELS_DIR: Path = REPO_ROOT / "models"
MLRUNS_DIR: Path = REPO_ROOT / "mlruns"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"


# ---------------------------------------------------------------------------
# Environment-driven settings
# ---------------------------------------------------------------------------


class EnvSettings(BaseSettings):
    """Settings sourced from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Runtime
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Data
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR

    # Tracking
    mlflow_tracking_uri: str = f"file:{MLRUNS_DIR.as_posix()}"

    # Persistence (later phases)
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/financial_ml"

    # API (later phases)
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# ---------------------------------------------------------------------------
# YAML configuration models
# ---------------------------------------------------------------------------


class _YamlModel(BaseModel):
    """Base for YAML-backed configuration sections."""

    model_config = ConfigDict(extra="forbid")


class AssetConfig(_YamlModel):
    """A single tradable asset.

    Phase 0 captures the schema only — ingestion is implemented in Phase 1.
    """

    symbol: str
    name: str = ""
    source: str = "yfinance"
    enabled: bool = True


class AssetsFile(_YamlModel):
    """Schema for ``configs/assets.yaml``."""

    assets: list[AssetConfig] = Field(default_factory=list)


class FeatureConfig(_YamlModel):
    """A single feature definition placeholder (Phase 3 fills this in)."""

    name: str
    enabled: bool = True
    params: dict = Field(default_factory=dict)


class FeaturesFile(_YamlModel):
    """Schema for ``configs/features.yaml``."""

    features: list[FeatureConfig] = Field(default_factory=list)


class ModelConfig(_YamlModel):
    """A single model entry placeholder (Phase 5+ fills this in)."""

    name: str
    enabled: bool = True


class ModelsFile(_YamlModel):
    """Schema for ``configs/models.yaml``."""

    models: list[ModelConfig] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset configuration (Phase 3)
# ---------------------------------------------------------------------------


class TargetConfig(_YamlModel):
    """Target variable definition."""

    name: str = "next_return"
    description: str = "Next-period simple return"
    method: Literal["simple", "log"] = "simple"
    horizon: int = 1


class SplitsConfig(_YamlModel):
    """Walk-forward split parameters."""

    train_window: int = 100
    test_window: int = 20
    gap: int = 0
    horizon: int = 1
    step: int = 1
    initial_train_days: int = 252


class ScalingConfig(_YamlModel):
    """Feature scaling configuration."""

    type: Literal["standard", "minmax"] = "standard"
    features_to_scale: list[str] = Field(default_factory=list)
    exclude_features: list[str] = Field(default_factory=list)


class OutputConfig(_YamlModel):
    """Dataset output configuration."""

    directory: str = "data/dataset/"
    train_prefix: str = "train"
    val_prefix: str = "val"
    test_prefix: str = "test"
    metadata_file: str = "dataset_metadata.json"
    file_format: Literal["parquet", "csv"] = "parquet"


class MetadataConfig(_YamlModel):
    """Metadata persistence configuration."""

    include_feature_schema: bool = True
    include_split_dates: bool = True
    include_scaler_params: bool = True
    include_target_definition: bool = True
    hash_configuration: bool = True


class ValidationConfig(_YamlModel):
    """Data quality validation configuration."""

    min_train_samples: int = 50
    max_test_samples: int = 100
    require_consecutive_timestamps: bool = True
    forbid_future_leakage: bool = True


class DatasetConfig(_YamlModel):
    """Schema for a single dataset configuration section."""

    target: TargetConfig = Field(default_factory=TargetConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


class DatasetsFile(_YamlModel):
    """Schema for ``configs/dataset.yaml``."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Bundle of all configuration sections."""

    env: EnvSettings
    assets: AssetsFile
    features: FeaturesFile
    models: ModelsFile
    dataset: DatasetsFile

    model_config = ConfigDict(arbitrary_types_allowed=True)


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """Load and cache the full application configuration.

    Called once per process. Tests can call :func:`load_config.cache_clear`
    to pick up edits to YAML files.
    """
    env = EnvSettings()
    assets = _load_yaml(CONFIG_DIR / "assets.yaml", AssetsFile)
    features = _load_yaml(CONFIG_DIR / "features.yaml", FeaturesFile)
    models = _load_yaml(CONFIG_DIR / "models.yaml", ModelsFile)
    dataset = _load_yaml(CONFIG_DIR / "dataset.yaml", DatasetsFile)
    return AppConfig(env=env, assets=assets, features=features, models=models, dataset=dataset)


def _load_yaml(path: Path, model: type[_YamlModel]) -> _YamlModel:
    """Load ``path`` as YAML and validate against ``model``.

    Missing files yield a default-constructed instance so the skeleton
    runs even before ``configs/`` is populated.
    """
    import yaml  # local import: PyYAML is heavy and not needed for type checks

    if not path.exists():
        return model()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return model.model_validate(data)


def reset_config_cache() -> None:
    """Clear the memoized configuration (test helper)."""
    load_config.cache_clear()
