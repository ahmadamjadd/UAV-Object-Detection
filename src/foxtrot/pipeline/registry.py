"""Fixed, code-owned adapter selection. Job specs can never name a module."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .contracts import Evaluator, Trainer
from .fixture import FixtureEvaluator, FixtureTrainer
from .ultralytics_adapter import UltralyticsEvaluator, UltralyticsTrainer


# Trusted worker settings: which adapter runs jobs, plus that adapter's own options.
class PipelineSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, validate_default=True)

    adapter: Literal["fixture", "ultralytics"]
    fixture_epoch_seconds: float = Field(default=0.05, ge=0.0, le=60.0, allow_inf_nan=False)
    fixture_fail_at_epoch: int | None = Field(default=None, ge=1)
    ultralytics_initialization: Literal["pretrained", "scratch"] = "pretrained"
    ultralytics_workers: int = Field(default=2, ge=0, le=16)


# The trainer and evaluator a job will use, and whether they are only a fixture.
@dataclass(frozen=True)
class PipelineAdapters:
    trainer: Trainer
    evaluator: Evaluator
    fixture: bool


# Build the adapters named by the trusted settings; nothing else can be chosen.
def get_adapters(settings: PipelineSettings) -> PipelineAdapters:
    if settings.adapter == "fixture":
        return PipelineAdapters(
            FixtureTrainer(epoch_seconds=settings.fixture_epoch_seconds,
                           fail_at_epoch=settings.fixture_fail_at_epoch),
            FixtureEvaluator(),
            fixture=True,
        )
    if settings.adapter == "ultralytics":
        return PipelineAdapters(
            UltralyticsTrainer(initialization=settings.ultralytics_initialization,
                               workers=settings.ultralytics_workers),
            UltralyticsEvaluator(),
            fixture=False,
        )
    raise ValueError(f"Unsupported pipeline adapter: {settings.adapter}")
