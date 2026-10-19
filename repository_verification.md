# Verification of repository issues 3–7

Date: **10 October 2026**. Initial scope: verify the five concerns in `research.md` through focused execution tests. Follow-up scope: implement and test fixes after user authorization.

**Reading this report:** issue sections 3–7 preserve the original findings before remediation. Their source line numbers and “no fix was applied” statements refer to that original investigation. The remediation section below records the corrected behavior; the original findings are retained for comparison.

## How to interpret these findings

These tests distinguish **a problem reproduced under a specific condition** from **a problem currently affecting your default dataset**. A reproducible bug is not automatically the explanation for poor training results. No YOLO training or model-accuracy comparison was performed.

One separate subagent investigated each issue, in batches to respect the available concurrency limit. Tests used copied source/resources and small fixtures under `/tmp`; the original source algorithms were not edited. Where needed, wrappers recorded intermediate values or controlled input ordering. These interventions are identified below.

Baseline inspected: Git revision `030c45fa75fb272bb6d0c0641209a9775d7125d0`; the existing research file was untracked. Runtime: Python 3.12.3, OpenCV 5.0.0, NumPy 2.5.3, Pillow 12.3.0, Ultralytics 8.4.174. No dependencies were installed and no hosted services were contacted for these tests.

**Current input inventory:** all 397 files in `data/raw/` are lowercase `.png`, all are 320×320. Existing generated images inspected in a 20-image sample are also 320×320. Square backgrounds are important because they hide width/height mistakes.

## Issue 3: Does the generator produce different datasets despite the seed?

**Verdict: background-order sensitivity is confirmed. Ordinary repeated-run nondeterminism was not reproduced. The import-time random saturation default does not affect the current generation path.**

### Test and observations

Five fresh Python processes used copies of three real backgrounds (`1.png`, `10.png`, `100.png`), current YAML with `n` reduced to 3 per class, and the unchanged generator's `load_params()` and `main()`. Each produced six images and six labels.

Two cases kept normal directory enumeration. Two supplied sorted enumeration for the background directory. One supplied reverse-sorted enumeration. The ordering intervention affected only that directory; other directory reads were left alone. Terminal clearing was suppressed. Wrappers recorded background reads and actual saturation arguments, and every output file was hashed.

| Comparison | Changed images | Changed labels | Meaning |
| --- | ---: | ---: | --- |
| Natural-order repeat | 0/6 | 0/6 | Same fixture and order reproduced exactly |
| Sorted-order repeat | 0/6 | 0/6 | Same controlled order reproduced exactly |
| Sorted versus reversed order | 5/6 | 0/6 | Ordering alone changed actual dataset images |

The unchanged image used the same middle background under both orders. Eleven background reads occurred because drawing attempts sometimes retry; all six expected images were ultimately written.

The pre-import random seed was deliberately changed from 1 to 2. This changed `saturate()`'s stored default from −8 to −10, but repeated outputs remained byte-for-byte identical. Every actual generation call supplied an explicit seeded saturation value: `[-8, -7, -3, -4, -6, -10]`. The random default was bypassed.

Aggregate SHA-256 of the filename-to-file-hash manifest:

```text
natural_a / natural_b / reverse:
eb7f08dbb22463d33d02aa03206046a42b5fc3feba42629d7c0f15ec67b63750

sorted_a / sorted_b:
eb7ea13f9ae9712324850a4b319f6c20cb6fa0c3757ec2c246a23a8777c90404
```

### Cause and practical impact

The seed repeats random indices, but the same index refers to a different file if the file list changes order. Background choices at [DataSetGeneration.py:531](src/data_generation/DataSetGeneration.py#L531) use unsorted `os.listdir()`. Seeding at line 303 does not impose file ordering.

**In easy terms:** running the small experiment twice worked consistently. Rebuilding the folder or using another computer could change which backgrounds are chosen, even with identical files and settings. The test deliberately changed enumeration to prove this dependency; it did not demonstrate spontaneous reordering in your current checkout.

The earlier import-time randomness concern needs narrowing. The default at line 142 looks suspicious, but line 655 supplies the value explicitly, so it is **not a demonstrated cause of changing datasets today**.

Potential future work: use a validated, sorted background manifest and record input/output hashes. No fix was applied. Cross-platform/library reproducibility, shadow settings and other cache states were not comprehensively tested.

### Reproduction and evidence

```bash
venv/bin/python /tmp/foxtrot-verify-issue3/verify.py
```

Evidence: `/tmp/foxtrot-verify-issue3/comparisons.json`, the five case manifests, and `findings.md`. This command recreates only the harness's temporary fixture directories.

## Issue 4: Does YAML `size` actually resize generated images?

**Verdict: a silent size mismatch is confirmed when the requested size differs from the background dimensions. The default currently matches because the backgrounds already measure 320×320.**

### Test and observations

The unchanged script entry point was executed in isolated copied projects, with `n=2` per class. Three cases used the same fixed 640×640 synthetic background and different YAML sizes. A fourth used a copy of the actual repository background `data/raw/1.png`.

| Input background, width×height | YAML `size` | Saved detection images | Generated letter resources | Exit code |
| --- | ---: | --- | --- | ---: |
| 640×640 | 160 | 4 images, all 640×640 | 36 resources, 160×160 | 0 |
| 640×640 | 320 | 4 images, all 640×640 | 36 resources, 320×320 | 0 |
| 640×640 | 640 | 4 images, all 640×640 | 36 resources, 640×640 | 0 |
| Actual 320×320 background | 640 | 4 images, all 320×320 | Resources 640×640 | 0 |

### Cause and practical impact

The YAML loader assigns `SIZE` and updates letter resources at lines 313 and 352. The main loop reads a background directly at line 543 and saves the resulting image at line 666 without resizing it. A resizing branch exists in the older CLI path, but the current entry point calls the YAML loader instead.

`size` is **not completely ignored**: it changes letter-resource resolution, sets the conditional crop size and can alter image content/retries. All tested normal entry-point runs produced zero letter crops because that mode is disabled by default. The setting nevertheless does not establish the saved detection-image dimensions.

**In easy terms:** choosing `size: 640` with your existing backgrounds still creates 320×320 detection images. The program can finish successfully while doing something different from what a user expects.

This does not show a default pipeline crash. It confirms that the parameter should not be presented as guaranteed output image resolution until its semantics are corrected or clarified. YOLO's training `imgsz` is a separate setting; `src/train.py` does not explicitly forward it.

Potential future work: define separate source/output/crop/training dimensions and apply/validate the chosen behavior. No fix was applied.

### Reproduction and evidence

```bash
venv/bin/python /tmp/foxtrot-verify-issue4/verify.py
venv/bin/python /tmp/foxtrot-verify-issue4/verify_current_background.py
```

Evidence: `/tmp/foxtrot-verify-issue4/results.json`, `current-background-results.json`, and each copied case's configuration, execution log, images and letter assets.

## Issue 5: Are labels wrong on rectangular backgrounds?

**Verdict: confirmed for rectangular inputs. Not reproduced on the square control, so this specific defect does not explain poor training on the existing square dataset by itself.**

### Test and observations

The real copied generator ran with one constant-color background per fixture, seed 42 and 20 images per class. Letters and noise were disabled and blur set to zero to isolate geometry. A wrapper called the original `draw_shape()` and captured its returned box plus the actual changed pixels; shape code, placement and label writing were unchanged.

| Image dimensions, width×height | Incorrect normalization | Raw boxes outside image | Entirely invisible shapes still labeled |
| --- | ---: | ---: | ---: |
| 320×320 | 0/40 | 0/40 | 0/40 |
| 640×320 | 40/40 | 30/40 | 11/40 |
| 320×640 | 40/40 | 27/40 | 11/40 |

All unclipped boxes matched independent rendered-pixel measurements within one pixel: 40 square, 10 wide and 13 tall examples. This establishes that the returned fields use ordinary horizontal x/width and vertical y/height coordinates. The problem is not a special coordinate convention.

### Concrete example

On a 640-wide, 320-high image, a visible Hexagon occupied approximately x=43–235 and y=49–157. Its raw box had center `(139.0841, 103.3408)`, width 192 and height 108.

```text
Correct normalized box: [0.2173189, 0.3229400, 0.3000000, 0.3375000]
Actually written box:  [0.4346378, 0.1614700, 0.6000000, 0.1687500]
```

A YOLO reader reconstructs center `(278.1682, 51.6704)`, width 384 and height 54 from the written label. The box moves away from the real shape and changes proportions.

### Cause and practical impact

Lines 644–645 divide x and width by image height, and y and height by image width. Square inputs hide this because both denominators are equal.

There is a second confirmed problem: lines 592–595 sample horizontal placement using image height and vertical placement using image width. This can move shapes partly or completely outside rectangular pictures while still writing a label. Fixing only the normalization would leave placement problems. Dimension construction and other shape/letter branches also deserve a coordinated review; they were not comprehensively tested here.

**In easy terms:** the program uses the vertical ruler to describe horizontal positions and vice versa. Rectangular inputs therefore produce misleading training labels, sometimes for a shape that is not visible at all.

These counts describe the specific fixtures, not expected frequencies for every dataset. They prove bad geometry/labels, not a measured decrease in mAP. No model was trained.

### Reproduction and evidence

```bash
MPLCONFIGDIR=/tmp/foxtrot-verify-issue5/mpl venv/bin/python /tmp/foxtrot-verify-issue5/verify.py
```

Evidence: `/tmp/foxtrot-verify-issue5/results.json`, `findings.md` and the saved generated datasets under `results/{square,wide,tall}/generated/`.

## Issue 6: Can background selection crash or hang?

**Original verdict: confirmed for problematic inputs; current 397 backgrounds are all OpenCV-readable lowercase PNG files.**

Seven isolated cases ran the original copied generator with `n: 1` (two expected images). A marker confirmed configuration loading completed, and a stack dump three seconds later identified the selection loop. Timeout cases were stopped after 15 seconds, not left running.

| Input | Original result |
| --- | --- |
| Readable `.png` | Success; two images |
| Readable `.png` plus `.txt` | Success; two images |
| Empty directory | `IndexError: Cannot choose from an empty sequence` |
| Only `.txt` files | Timeout; stuck choosing unsupported files |
| Corrupt `.jpg` | `AttributeError` after `cv2.imread()` returned `None` |
| Readable uppercase `.PNG` only | Timeout; extension rejected |
| Readable `.jpeg` only | Timeout; extension rejected |

The original inner selection loop had no retry bound; the outer drawing-attempt limit did not protect it. Empty inputs failed before rendering, and accepted but unreadable files failed when accessing image dimensions.

In easy terms: the program did not first check whether it had usable backgrounds. It could keep looking forever or crash with an unrelated-looking error. Original evidence and harness: `/tmp/foxtrot-verify-issue6/{verify.py,child.py,results.json}` and the case logs. Reproduction command for the retained original fixtures: `venv/bin/python -B /tmp/foxtrot-verify-issue6/verify.py`.

## Issue 7: Can generator dependencies prevent training from starting?

**Original verdict: conditional import failure confirmed; import works in the complete current environment. Dataset deletion on import was not reproduced.**

Six isolated cases imported copied `src/train.py` without calling training. Genuine training dependencies were loaded first; environment-file loading was disabled and network connections blocked. Faults were introduced only into generator-related dependencies or copied resources.

| Case | Original training-module import |
| --- | --- |
| Complete environment/resources | Success; classes Hexagon and Triangle |
| Simulated missing `seaborn` | Failed at generator import |
| Simulated missing `resize_backgrounds` | Failed at generator import |
| Missing generator `vertex.json` | `FileNotFoundError` |
| Malformed generator `vertex.json` | `JSONDecodeError` |
| Simulated missing Matplotlib | Failed through Seaborn; Matplotlib is also a YOLO dependency, so this is not uniquely generator-related |

Training imported the generator merely to get class names. That import nevertheless loaded plotting/generation modules, initialized blur kernels and read shape vertices. All copied-file hashes and dataset sentinel files remained unchanged: it did **not** call generation or delete datasets.

In easy terms: training an existing dataset unnecessarily required parts of the dataset generator to be present and healthy. The import coupling was real; the current complete installation was not shown to be broken. Original evidence: `/tmp/foxtrot-verify-issue7/{verify.py,results.json,findings.md}`. Command: `venv/bin/python /tmp/foxtrot-verify-issue7/verify.py`. The harness catches expected exceptions, so its process exit code alone does not indicate whether a particular import passed; consult the recorded import markers.

The original `/tmp` harnesses refer to retained temporary fixtures and may disappear after the session. The repository regression tests added during remediation are the lasting checks for corrected behavior.

## Remediation: implemented fixes and verification

The user subsequently authorized fixes for all five issues. A separate subagent handled each issue; shared generator edits were applied in sequence and reviewed together by the parent agent.

| Issue | Implemented correction | Lasting verification |
| --- | --- | --- |
| 3: reproducibility | Sorted validated background paths, sorted lighting/glyph enumeration, cleared stale glyph masks, sampled optional saturation at call time after seeding | Three fresh processes with different pre-import random states and deliberately reversed actual background enumeration produce identical hashes for six images and six labels each |
| 4: image dimensions | Positive-integer output size; background resized before drawing; letter cache checked for completeness/readability/dimensions and rebuilt when needed | Real script outputs exact 160×160, 320×320 and 640×640 images from mixed input dimensions; invalid size preserves previous outputs; incomplete/wrong-size glyph cache rebuilds all 36 resources with letters always enabled |
| 5: geometry/labels | Correct x/width versus y/height calculations, placement bounds, raster-derived boxes including borders, letter positioning/crop axes and circle sectors; rejected clipping and bounded failed-generation attempts | 144 polygon rotations and 60 actual writer labels across square/wide/tall canvases; rendered pixels independently compared to boxes; letter/circle/border/clipping and oversized-input checks |
| 6: background robustness | Readable PNG/JPG/JPEG discovery with case-insensitive extensions; skip corrupt files with bounded warnings; clear errors for unusable inputs before clearing prior outputs; detect files corrupted/removed after discovery | Helper tests plus real entry-point tests for empty/unsupported/corrupt/missing input, output preservation and successful mixed uppercase/JPEG input |
| 7: training import | Lightweight immutable shared class metadata; training no longer imports generation modules/resources; credential-file loading moved inside explicit training entry point | Genuine YOLO dependency with generator modules blocked and no generator resources; package/direct-script imports and exact YAML class mapping verified; no import-time `.env` reads |

Source additions: `src/class_names.py`, `src/data_generation/backgrounds.py`, `src/data_generation/image_size.py`. Generation and training both use the shared class order. DVC dependencies include the new class metadata so changes invalidate the appropriate pipeline stages.

### Combined regression result

**All 21 tests passed in 49.102 seconds** against the integrated implementation:

```bash
MPLCONFIGDIR=/tmp/foxtrot-final-tests-mpl \
YOLO_CONFIG_DIR=/tmp/foxtrot-final-tests-yolo \
venv/bin/python -B -m unittest discover -s tests -v
```

Tests are retained in `tests/test_backgrounds.py`, `test_determinism.py`,
`test_generator_preflight.py`, `test_geometry.py`, `test_image_size.py` and
`test_training_import.py`. Generation tests use temporary output paths; entry-point
tests use copied source/resources. A SHA-256 comparison of **1,222 protected files**
confirmed no changes to existing data, training outputs, `params.yaml`, `data.yaml`
or `dvc.lock`, and no new files in the original `data/` or `runs/` directories.
Source whitespace checks also passed with the original CRLF line endings accounted for.

### End-to-end generation and split check

An isolated copy used **all 397 actual background files and the unchanged project YAML**. The corrected generator and split script both completed successfully:

- 200 generated images and 200 labels, all images 320×320.
- Exactly 100 Hexagon and 100 Triangle labels.
- All 200 boxes had positive dimensions and stayed within normalized image bounds.
- Split outputs contained 160 training images and 40 validation images.
- All 36 letter resources were generated in the temporary copy.

The acceptance fixture and recorded summary are at `/tmp/foxtrot-default-acceptance-ogvko27e/acceptance_summary.json`. This checks real generation/splitting, not model accuracy.

### Behavior changes and limits

`size` now means a square output canvas; rectangular source backgrounds are stretched to that canvas. Geometry tests intentionally preserve rectangular canvases to verify the coordinate fix independently rather than letting square resizing hide it. Oversized/impossible shape settings now fail explicitly instead of silently returning an incomplete dataset.

The old `letters320` resource folder was empty. Testing with letters always enabled exposed this; checking only folder existence was insufficient. The loader now checks every expected glyph's readability and dimensions before reusing the cache.

Sorting, resizing and corrected rendered-pixel box bounds can change generated outputs relative to older code. Existing datasets/checkpoints are not rewritten automatically. The tests do not promise identical results across different library versions/hardware, and no GPU training, MLflow upload or model-accuracy evaluation was performed.

Issues 1–2 and 8, the website architecture, broader parameter coverage and hyperparameter tuning remain outside this remediation request.

