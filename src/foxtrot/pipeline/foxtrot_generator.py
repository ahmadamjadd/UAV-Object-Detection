"""Run the original Foxtrot generator with explicit inputs (V2).

The legacy generator in ``src/data_generation`` reads params.yaml, writes into
``data/generated`` (deleting what was there) and caches letter images inside its
own ``res/`` folder. Those files are protected, so this wrapper never edits
them. Instead it loads the module in this (separate) process and, before
calling its ``main()``, points its settings and folders at explicit values:

* settings come from a validated :class:`GenerationRecipe`;
* backgrounds are an exact, already-checked list of files;
* images/labels go to a folder the caller chose (a temporary build folder);
* letter images are cached in the workspace instead of ``res/``.

Two small stand-ins observe the run without changing what it draws: one for
``cv2`` notices each saved image (progress, cancel, which background it used),
one for ``os`` stops the terminal from being cleared.
"""

import contextlib
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import random
import shutil
import sys
from uuid import uuid4
import warnings

from foxtrot.config.generation import GenerationRecipe

from .contracts import PipelineCancelled, PipelineFailure, ProgressReporter

GENERATOR_ENV = "FOXTROT_GENERATOR_DIR"  # tests may point at a copy of the generator
BACKGROUND_EXTENSIONS = (".png", ".jpg", ".jpeg")


# Raised when the original generator code cannot be found next to this package.
class GeneratorUnavailable(PipelineFailure):
    # The generator code could not be found or loaded.
    def __init__(self, message: str):
        super().__init__("generator_unavailable", message)


# What one generator run produced: class order and, per image file, its background photo.
@dataclass(frozen=True)
class GeneratedImages:
    class_names: tuple[str, ...]
    sources: dict[str, str]  # "12.jpg" -> "background-7.png"


# Where the original generator lives: src/data_generation in a source checkout.
def generator_directory() -> Path:
    override = os.environ.get(GENERATOR_ENV)
    return Path(override) if override else Path(__file__).resolve().parents[2] / "data_generation"


# The files whose content decides what the generator draws (used for dataset identity).
def generator_files() -> list[Path]:
    directory = generator_directory()
    if not (directory / "DataSetGeneration.py").is_file():
        raise GeneratorUnavailable(
            f"The Foxtrot generator was not found at {directory}. Run Foxtrot from the source checkout.")
    files = sorted(directory.glob("*.py"))
    resources = directory / "res"
    files += [resources / "vertex.json", resources / "ArielBold.ttf"]
    files += sorted(p for p in (resources / "lights").iterdir() if p.is_file())
    files.append(directory.parent / "class_names.py")
    return files


# Sorted background images in a folder; each must decode, so problems show before any work.
def list_backgrounds(folder: str | Path) -> list[Path]:
    import cv2

    root = Path(folder)
    if not root.is_absolute() or not root.is_dir():
        raise PipelineFailure("backgrounds_missing", f"Backgrounds folder not found: {folder}")
    found, unreadable = [], []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in BACKGROUND_EXTENSIONS:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        (found if image is not None and image.size else unreadable).append(path)
    if unreadable:
        names = ", ".join(p.name for p in unreadable[:5])
        raise PipelineFailure("backgrounds_unreadable", f"{len(unreadable)} background image(s) cannot be read: {names}")
    if not found:
        raise PipelineFailure("backgrounds_missing", f"No PNG or JPEG background images in {root}")
    return found


# Import the original generator module once per process (it keeps settings in globals).
def _load_legacy_module():
    directory = generator_directory()
    generator_files()  # clear error if it is missing
    for path in (str(directory.parent), str(directory)):
        if path not in sys.path:
            sys.path.insert(0, path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)  # it opens vertex.json without closing it
        module = importlib.import_module("DataSetGeneration")
    import colorama
    colorama.deinit()  # its import wrapped sys.stdout for terminal colours; undo that
    return module


# True if every letter image exists at exactly this canvas size.
def _letters_complete(folder: Path, names, size: int) -> bool:
    import cv2

    for name in names:
        path = folder / f"{name}.png"
        image = cv2.imread(str(path)) if path.is_file() else None
        if image is None or image.shape[:2] != (size, size):
            return False
    return True


# Make (once) the letter images for this canvas size in the workspace cache, never in res/.
# They are built in a temporary folder and renamed, so a half-written cache is never used.
def ensure_letters(module, cache_root: Path, size: int) -> Path:
    target = cache_root / f"letters-{size}"
    if _letters_complete(target, module.LETTER_CLASS, size):
        return target
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache_root / f".letters-{size}-{uuid4().hex}"
    module.genl.generate_letters(size, size, str(temporary))
    if target.exists():  # an older incomplete cache; replace it
        shutil.rmtree(target, ignore_errors=True)
    try:
        temporary.rename(target)
    except OSError:  # another run finished first; use theirs
        shutil.rmtree(temporary, ignore_errors=True)
    if not _letters_complete(target, module.LETTER_CLASS, size):
        raise PipelineFailure("letters_failed", "Letter images could not be prepared")
    return target


# Stand-in for the generator's "os": identical, except clearing the terminal does nothing.
class _QuietOs:
    # Everything else behaves like the real os module.
    def __getattr__(self, name):
        return getattr(os, name)

    # The generator runs os.system('clear'); skip it.
    @staticmethod
    def system(_command):
        return 0


# Stand-in for the generator's "cv2": identical, but tells us about every image it saves.
class _WatchedCv2:
    # Wrap OpenCV so every saved image is reported.
    def __init__(self, cv2_module, on_image):
        self._cv2, self._on_image = cv2_module, on_image

    # Everything else behaves like the real OpenCV module.
    def __getattr__(self, name):
        return getattr(self._cv2, name)

    # Save as usual, fail loudly if saving failed, then report the saved file.
    def imwrite(self, path, image, *arguments):
        if not self._cv2.imwrite(path, image, *arguments):
            raise PipelineFailure("write_failed", f"Could not save generated image {Path(path).name}")
        self._on_image(Path(path))
        return True


# Copy the recipe into the generator's settings (the same names load_params() would set).
def _apply_recipe(module, recipe: GenerationRecipe) -> None:
    import cv2

    module.N = recipe.images_per_class
    module.SIZE = module.validate_image_size(recipe.canvas_size)
    module.L_SIZE = int(0.4 * recipe.canvas_size)
    module.SHAPE_SCALE = [recipe.shape_scale.low, recipe.shape_scale.high]
    module.NOISE_FACTOR = recipe.noise
    module.BLUR_FACTOR = [recipe.blur.low, recipe.blur.high]
    module.BRIGHTNESS_DIFFERENCE = recipe.brightness_difference
    module.ALPHA = [recipe.contrast.low, recipe.contrast.high]
    module.BETA = [recipe.brightness.low, recipe.brightness.high]
    module.SATURATION = [recipe.saturation.low, recipe.saturation.high]
    module.DESATURATE = recipe.desaturate
    module.ADD_LETTER = recipe.add_letter
    module.LETTER_PROB = recipe.letter_probability
    module.ADD_LIGHT_GRADIENT = recipe.add_light_gradient
    module.LIGHT_PROB = recipe.light_probability
    module.OPACITY = [0.0, recipe.light_opacity]
    module.KERNEL_CACHE = {max(1, i): cv2.getGaussianKernel(max(1, i), 0)
                           for i in range(recipe.blur.low, recipe.blur.high + 1)}
    # Terminal-only features stay off; the letter-crop dataset is not part of V2.
    module.DISPLAY = module.SAVE_BOUNDS = module.SHOW_DISTRIBUTION = False
    module.LETTER_DATASET = module.ONLY_LETTER = False
    module.FREQ = 0


# Generate images/ and labels/ inside `output` (which must not exist yet).
# Same recipe + same backgrounds + same generator code = the same pictures.
def generate_images(
    recipe: GenerationRecipe, backgrounds: list[Path], output: Path, letters_cache: Path,
    reporter: ProgressReporter,
) -> GeneratedImages:
    import cv2
    import numpy as np

    module = _load_legacy_module()
    images, labels = output / "images", output / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    _apply_recipe(module, recipe)
    module.Letters = str(ensure_letters(module, letters_cache, recipe.canvas_size))
    module.Background, module.Output, module.Labels = str(backgrounds[0].parent), str(images), str(labels)
    unused = str(output / "unused")  # letter-crop outputs; never written with the settings above
    module.LetterOutput = module.LetterLabels = module.Bounds = module.LetterBounds = unused
    module.discover_backgrounds = lambda _folder: [str(path) for path in backgrounds]

    current = {"background": None}
    original_read = module.read_background

    # Remember which background the generator opened last; the next saved image used it.
    def read_background(path):
        current["background"] = Path(path).name
        return original_read(path)

    class_names = tuple(module.SHAPES)
    total = recipe.images_per_class * len(class_names)
    every = max(1, total // 50)
    sources: dict[str, str] = {}

    # Called for every saved image: record its background, report progress, honour cancel.
    def on_image(path: Path):
        sources[path.name] = current["background"]
        if len(sources) % every == 0 or len(sources) == total:
            reporter.progress({"phase": "generating", "done": len(sources), "total": total})
        if reporter.wait_cancelled():
            raise PipelineCancelled()

    module.read_background = read_background
    module.cv2, module.os = _WatchedCv2(cv2, on_image), _QuietOs()
    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    cv2.setRNGSeed(recipe.seed)
    with contextlib.redirect_stdout(sys.stderr):  # its progress printing goes to the log
        module.main()
    label_count = sum(1 for _ in labels.glob("*.txt"))
    if len(sources) != total or label_count != total:
        raise PipelineFailure("generation_incomplete",
                              f"Expected {total} images, got {len(sources)} images and {label_count} labels")
    return GeneratedImages(class_names=class_names, sources=sources)
