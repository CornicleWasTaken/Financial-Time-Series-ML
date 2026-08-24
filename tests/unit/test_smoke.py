"""Smoke tests.

These exist purely to confirm the package skeleton is importable and the
configuration layer loads without errors. They run as part of ``make test``
and are the only tests required to be passing at the end of Phase 0.
"""

from __future__ import annotations

from financial_ml import __version__
from financial_ml.config import CONFIG_DIR, DATA_DIR, REPO_ROOT, load_config


def test_package_has_version() -> None:
    assert isinstance(__version__, str) and __version__


def test_repo_paths_are_absolute() -> None:
    assert REPO_ROOT.is_absolute()
    assert DATA_DIR.is_absolute()
    assert CONFIG_DIR.is_absolute()


def test_config_loads_end_to_end(app_config) -> None:
    cfg = load_config()
    assert cfg.assets.assets, "configs/assets.yaml should seed the asset universe"
    assert cfg.env.app_env in {"dev", "test", "prod"}


def test_assets_yaml_has_at_least_one_enabled(app_config) -> None:
    enabled = [a for a in load_config().assets.assets if a.enabled]
    assert enabled, "Phase 1 needs at least one enabled asset to ingest"
