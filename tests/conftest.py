from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from fuellake.config import Settings
from fuellake.pipeline import RunParams, build_pipeline
from tests.synth import SynthConfig, write_monthly_files

# "today" for the synthetic dataset: right after its last week (2023-01-02 + 129 weeks = 2025-06-23)
AS_OF = date(2025, 7, 1)
FAST_MODELS = ("naive", "ma4", "ridge", "gbm", "blend")
HORIZONS = (1, 2, 4)
# heavy sampling noise: the weekly state average is a poor measure of the level, so there is real structure to learn
NOISY = SynthConfig(station_noise=0.35)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("synthetic_raw")
    write_monthly_files(directory, NOISY)
    return directory


@pytest.fixture(scope="session")
def lake(tmp_path_factory, synth_dir) -> Settings:
    """A fully built lake (bronze -> forecast -> report) on synthetic data, built once per test session."""
    settings = Settings(tmp_path_factory.mktemp("lake"))
    params = RunParams(local_dir=synth_dir, models=FAST_MODELS, horizons=HORIZONS, n_folds=12, as_of=AS_OF)
    run = build_pipeline(settings, params, log=lambda message: None).run(run_dir=settings.runs)
    assert run.ok, {name: r.error for name, r in run.results.items() if r.error}
    return settings


@pytest.fixture
def empty_lake(tmp_path) -> Settings:
    settings = Settings(tmp_path / "lake")
    settings.ensure()
    return settings
