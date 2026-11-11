"""Fixed, code-owned adapter selection. Job specs can never name a module."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .contracts import Evaluator, Trainer
from .fixture import FixtureEvaluator, FixtureTrainer


class PipelineSettings(BaseModel):
    """Trusted worker settings. V1 adds the real Ultralytics adapter name."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, validate_default=True)

    adapter: Literal["fixture"]
    fixture_epoch_seconds: float = Field(default=0.05, ge=0.0, le=60.0, allow_inf_nan=False)
    fixture_fail_at_epoch: int | None = Field(default=None, ge=1)


@dataclass(frozen=True)
class PipelineAdapters:
    trainer: Trainer
    evaluator: Evaluator
    fixture: bool


def get_adapters(settings: PipelineSettings) -> PipelineAdapters:
    if settings.adapter == "fixture":
        return PipelineAdapters(
            FixtureTrainer(epoch_seconds=settings.fixture_epoch_seconds,
                           fail_at_epoch=settings.fixture_fail_at_epoch),
            FixtureEvaluator(),
            fixture=True,
        )
    raise ValueError(f"Unsupported pipeline adapter: {settings.adapter}")
