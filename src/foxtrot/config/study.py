"""Tuning-study definitions (V5 grid/random; V6 adaptive, model comparison, recipes, repeats).

A study runs several *trials*. Every trial resolves to an ordinary, complete
training job (same validation as a hand-made job), so a trial can never use a
setting a normal job could not. Search parameters name a training setting
(e.g. ``lr0``), or a dataset-recipe setting (``recipe.noise``), or the low end
and span of a recipe range (``recipe.shape_scale.low`` / ``.span``), which is
the constrained way to tune a range: the high end is low + span, so every
sampled pair is valid.

Generator ranges (variation *within* one dataset, e.g. shape_scale 0.2–0.6) and
tuning ranges (choices *between* trials) are separate: a tuning parameter picks
the generator range a trial's dataset uses.
"""

import itertools
import math
import random
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .generation import GenerationRecipe, GenerationRequest
from .models import MODEL_IDS
from .training import TrainingSettings

Value = str | int | float | bool
OBJECTIVES = {  # objective -> (final score field in the evaluation, per-epoch progress key)
    "map50_95": ("map50_95", "metrics/mAP50-95"),
    "map50": ("map50", "metrics/mAP50"),
    "precision": ("precision", "metrics/precision"),
    "recall": ("recall", "metrics/recall"),
}
RECIPE_SCALARS = {"images_per_class", "noise", "brightness_difference", "letter_probability",
                  "light_probability", "light_opacity", "desaturate", "add_letter", "add_light_gradient"}
RECIPE_RANGES = {"shape_scale", "blur", "contrast", "brightness", "saturation"}
MAX_GRID = 500


# A strict, read-only base: unknown fields and wrong types are refused.
class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


# Sample this parameter only when another parameter (or fixed setting) has one of these values.
class Condition(_Strict):
    param: str = Field(min_length=1, max_length=80)
    values: tuple[Value, ...] = Field(min_length=1, max_length=50)


# One search dimension: a number range (optionally stepped or logarithmic) or a list of choices.
class SearchParam(_Strict):
    name: str = Field(min_length=1, max_length=80)
    type: Literal["int", "float", "categorical"]
    low: float | None = None
    high: float | None = None
    step: float | None = Field(default=None, gt=0)
    log: bool = False
    choices: tuple[Value, ...] | None = Field(default=None, min_length=1, max_length=50)
    condition: Condition | None = None

    # Ranges need low < high; choices need a list; log needs positive bounds and no step.
    @model_validator(mode="after")
    def shape(self) -> Self:
        if self.type == "categorical":
            if self.choices is None or self.low is not None or self.high is not None or self.log:
                raise ValueError(f"{self.name}: categorical parameters take only 'choices'")
            if len(set(map(repr, self.choices))) != len(self.choices):
                raise ValueError(f"{self.name}: choices must be different")
        else:
            if self.low is None or self.high is None or self.choices is not None:
                raise ValueError(f"{self.name}: ranges need 'low' and 'high' (and no choices)")
            if not self.low < self.high:
                raise ValueError(f"{self.name}: low must be below high")
            if self.log and (self.low <= 0 or self.step is not None):
                raise ValueError(f"{self.name}: logarithmic ranges need low > 0 and no step")
            if self.type == "int" and (self.low != int(self.low) or self.high != int(self.high)
                                       or (self.step is not None and self.step != int(self.step))):
                raise ValueError(f"{self.name}: integer ranges need whole numbers")
        return self

    # Every value a finite parameter can take (grid search); None if it is continuous.
    def values(self) -> list[Value] | None:
        if self.type == "categorical":
            return list(self.choices)
        if self.type == "int" and not self.log:
            step = int(self.step or 1)
            return list(range(int(self.low), int(self.high) + 1, step))
        if self.type == "float" and self.step is not None:
            count = int(math.floor((self.high - self.low) / self.step + 1e-9))
            return [round(self.low + index * self.step, 10) for index in range(count + 1)]
        return None

    # The two ends of the range (or every choice), used to check settings before the study starts.
    def extremes(self) -> list[Value]:
        if self.type == "categorical":
            return list(self.choices)
        values = self.values()
        if values:
            return [values[0], values[-1]]
        return [int(self.low), int(self.high)] if self.type == "int" else [self.low, self.high]


# One configuration to repeat under another training seed (V6 repeat workflow).
class RepeatItem(_Strict):
    model_id: Literal[*MODEL_IDS]
    training: dict[str, Value] = Field(default_factory=dict)
    seed: int = Field(ge=0, le=4_294_967_295)
    source_trial: int | None = None  # the trial of an earlier study it repeats


# A complete study request; everything is checked before the first trial starts.
class StudySpec(_Strict):
    name: str = Field(min_length=1, max_length=80)
    dataset_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    recipe: GenerationRequest | None = None  # V6: tune the dataset recipe (training data regenerated)
    validation_share: float = Field(default=0.2, ge=0.1, le=0.5)  # recipe studies: backgrounds kept for validation
    model_ids: tuple[Literal[*MODEL_IDS], ...] = Field(default=("yolov8n",), min_length=1, max_length=5)
    model_budget: Literal["equal", "combined"] = "equal"
    initialization: Literal["pretrained", "scratch"] = "pretrained"
    tracking_destination_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    base_training: dict[str, Value] = Field(default_factory=dict)
    search_space: tuple[SearchParam, ...] = Field(default=(), max_length=12)
    sampler: Literal["grid", "random", "tpe", "repeat"] = "random"
    repeats: tuple[RepeatItem, ...] = Field(default=(), max_length=60)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    trials_per_model: int = Field(default=5, ge=1, le=200)
    time_limit_minutes: int | None = Field(default=None, ge=1, le=10_080)
    max_epochs_per_trial: int = Field(default=50, ge=1, le=1000)
    objective: Literal[*OBJECTIVES] = "map50_95"
    failure_policy: Literal["continue", "stop"] = "continue"
    pruning: bool = False

    # Cross-field rules: a data source, known parameters, finite grids, valid combinations.
    @model_validator(mode="after")
    def check_study(self) -> Self:
        if (self.dataset_id is None) == (self.recipe is None):
            raise ValueError("Choose either a fixed dataset or a dataset recipe to tune")
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("Each model may be listed once")
        names = [p.name for p in self.search_space]
        if len(set(names)) != len(names):
            raise ValueError("Each search parameter may appear once")
        for param in self.search_space:
            _check_param_name(param, recipe_study=self.recipe is not None)
            if param.name in self.base_training:
                raise ValueError(f"{param.name} is searched, so it cannot also be a fixed setting")
            if param.condition is not None:
                earlier = names[:names.index(param.name)]
                if param.condition.param not in earlier and param.condition.param not in TrainingSettings.model_fields:
                    raise ValueError(f"{param.name}: its condition must refer to an earlier parameter or a setting")
        if self.sampler == "repeat":
            if not self.repeats or self.search_space:
                raise ValueError("A repeat study lists repeats and has no search space")
        elif self.repeats:
            raise ValueError("Repeats are only used by repeat studies")
        if self.sampler == "grid":
            if any(p.values() is None for p in self.search_space):
                raise ValueError("Grid search needs choices, integer ranges or stepped float ranges")
            if len(grid_points(self)) > MAX_GRID:
                raise ValueError(f"The grid has more than {MAX_GRID} combinations")
        if self.pruning and (self.sampler not in ("random", "tpe") or
                             (len(self.model_ids) > 1 and self.model_budget == "combined")):
            raise ValueError("Pruning needs random or TPE search, within one model at a time "
                             "(not a combined search across models)")
        self._check_combinations()
        return self

    # Try the fixed settings with the range ends / choices, so impossible combinations fail now.
    def _check_combinations(self) -> None:
        candidates = []
        if self.sampler == "repeat":
            candidates = [(item.model_id, dict(item.training, seed=item.seed)) for item in self.repeats]
        else:
            if self.sampler == "grid":
                combos = grid_points(self)
            else:
                ends = [[(p.name, v) for v in p.extremes()] for p in self.search_space]
                combos = [dict(c) for c in itertools.product(*ends)] if ends else [{}]
                if len(combos) > 256:
                    combos = random.Random(self.seed).sample(combos, 256)
            candidates = [(model, combo) for model in self.model_ids for combo in combos]
        for model_id, combo in candidates:
            active = active_params(self, combo)
            try:
                training, recipe = resolve_settings(self, active)
            except (ValidationError, ValueError) as error:
                detail = error.errors()[0]["msg"] if isinstance(error, ValidationError) else str(error)
                raise ValueError(f"Invalid combination {active or 'with the fixed settings'}: {detail}. "
                                 "Use a condition or change the ranges.") from None
            if training.epochs > self.max_epochs_per_trial:
                raise ValueError(f"epochs {training.epochs} exceeds the per-trial limit {self.max_epochs_per_trial}")


# Only names Foxtrot can really apply are allowed, with types matching the setting.
def _check_param_name(param: SearchParam, *, recipe_study: bool) -> None:
    if param.name.startswith("recipe."):
        if not recipe_study:
            raise ValueError(f"{param.name}: recipe parameters need a recipe study")
        parts = param.name.split(".")
        if len(parts) == 2 and parts[1] in RECIPE_SCALARS:
            return
        if len(parts) == 3 and parts[1] in RECIPE_RANGES and parts[2] in ("low", "span"):
            return
        raise ValueError(f"{param.name}: not a tunable recipe setting")
    field = TrainingSettings.model_fields.get(param.name)
    if field is None or param.name == "device":
        raise ValueError(f"{param.name}: not a tunable training setting")
    annotation = str(field.annotation)
    if param.type == "int" and "int" not in annotation:
        raise ValueError(f"{param.name}: is not a whole-number setting")
    if param.type == "float" and "float" not in annotation:
        raise ValueError(f"{param.name}: is not a decimal setting")


# Drop parameters whose condition is not met by the other values or the fixed settings.
def active_params(spec: StudySpec, values: dict[str, Value]) -> dict[str, Value]:
    known = {**spec.base_training}
    active = {}
    for param in spec.search_space:
        if param.name not in values:
            continue
        if param.condition is not None:
            current = {**TrainingSettings().model_dump(), **known, **active}.get(param.condition.param)
            if current not in param.condition.values:
                continue
        active[param.name] = values[param.name]
    return active


# Every grid combination (respecting conditions, without duplicates).
def grid_points(spec: StudySpec) -> list[dict[str, Value]]:
    axes = [[(p.name, v) for v in p.values()] for p in spec.search_space]
    seen, points = set(), []
    for combo in itertools.product(*axes) if axes else [()]:
        active = active_params(spec, dict(combo))
        key = repr(sorted(active.items()))
        if key not in seen:
            seen.add(key)
            points.append(active)
    return points


# How many trials the study will run at most (shown before starting).
def planned_trials(spec: StudySpec) -> int:
    if spec.sampler == "repeat":
        return len(spec.repeats)
    per_model = len(grid_points(spec)) if spec.sampler == "grid" else spec.trials_per_model
    return per_model * len(spec.model_ids) if spec.model_budget == "equal" else per_model


# The full training settings (and, for recipe studies, the recipe) a trial resolves to.
def resolve_settings(spec: StudySpec, params: dict[str, Value]) -> tuple[TrainingSettings, GenerationRecipe | None]:
    training = {**spec.base_training, **{k: v for k, v in params.items() if not k.startswith("recipe.")}}
    settings = TrainingSettings.model_validate(training)
    if spec.recipe is None:
        return settings, None
    recipe = spec.recipe.recipe.model_dump()
    for name, value in params.items():
        parts = name.split(".")
        if len(parts) == 2 and parts[0] == "recipe":
            recipe[parts[1]] = value
    for range_name in RECIPE_RANGES:  # low + span -> a valid (low, high) pair
        low, span = params.get(f"recipe.{range_name}.low"), params.get(f"recipe.{range_name}.span")
        if low is not None or span is not None:
            current = recipe[range_name]
            low = current["low"] if low is None else low
            span = (current["high"] - current["low"]) if span is None else span
            recipe[range_name] = {"low": low, "high": round(low + span, 10)}
    return settings, GenerationRecipe.model_validate(recipe)
