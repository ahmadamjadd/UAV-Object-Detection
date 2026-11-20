"""Divide generated images into train/ and val/ reproducibly (V2).

Unlike ``src/split_dataset.py`` (which reads params.yaml and replaces
``data/split``), this takes explicit folders and settings and writes only into
a new output folder. With grouping on, all images made from one background
photo go to the same split, so validation scores are not inflated by
backgrounds the model already saw during training.
"""

from dataclasses import dataclass
from pathlib import Path
import random
import shutil

from foxtrot.config.generation import SplitSettings

from .contracts import PipelineFailure


# Which images went where, and which background groups each split holds.
@dataclass(frozen=True)
class SplitOutcome:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]


# Decide the split without touching files. `sources` maps image name -> background name.
def plan_split(sources: dict[str, str], settings: SplitSettings) -> SplitOutcome:
    names = sorted(sources)
    if len(names) < 2:
        raise PipelineFailure("split_too_small", "At least 2 images are needed to make train and validation sets")
    rng = random.Random(settings.seed)
    target = round(len(names) * settings.train_ratio)
    if not settings.group_by_background:
        rng.shuffle(names)
        cut = min(max(target, 1), len(names) - 1)
        train, validation = names[:cut], names[cut:]
    else:
        groups: dict[str, list[str]] = {}
        for name in names:
            groups.setdefault(sources[name], []).append(name)
        if len(groups) < 2:
            raise PipelineFailure("split_one_background",
                                  "Grouped splitting needs images from at least 2 different backgrounds")
        order = sorted(groups)
        rng.shuffle(order)
        train, validation = [], []
        for group in order:  # fill train up to the target with whole groups; the rest is validation
            (train if not train or len(train) + len(groups[group]) <= target else validation).extend(groups[group])
        if not validation:  # every group fitted into train; give the last one to validation
            last = order[-1]
            train = [n for n in train if sources[n] != last]
            validation = list(groups[last])
    return SplitOutcome(
        train=tuple(sorted(train)), validation=tuple(sorted(validation)),
        train_groups=tuple(sorted({sources[n] for n in train})),
        validation_groups=tuple(sorted({sources[n] for n in validation})),
    )


# Copy images and labels into output/train/{images,labels} and output/val/{images,labels}.
def write_split(source: Path, output: Path, outcome: SplitOutcome) -> None:
    for split, names in (("train", outcome.train), ("val", outcome.validation)):
        images, labels = output / split / "images", output / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for name in names:
            shutil.copy2(source / "images" / name, images / name)
            label = f"{Path(name).stem}.txt"
            shutil.copy2(source / "labels" / label, labels / label)
