# H1 development baseline

This is the development foundation for the planned local application. It provides
an installable Python package and verified dependencies; it does not yet provide
a website, API server, worker, or a `foxtrot serve` command.

## Verified target and scope

- Ubuntu 24.04.4 LTS, Linux x86_64; Intel Core i5-13420H, CPU execution.
- CPython 3.14.4 (`.python-version`). Package metadata accepts Python 3.14 only.
  Switched from the H1 target CPython 3.12.3 on 11 November 2026 after the OS
  upgrade removed 3.12; all 186 locked versions are unchanged (see "Maintaining locks").
- Node 22.17.0 (`.nvmrc`) and npm 10.9.2, needed by developers for the frontend.
- The user chose this machine for now and deferred a second machine in favor of
  later Docker containerization. Docker is not implemented or verified in H1.
- GPU execution, other operating systems, and production installation remain
  unverified. No accelerator was exposed to this session. The standard PyTorch
  wheels include CUDA libraries and are large even when executing on CPU.

Direct Python dependencies are pinned in `pyproject.toml`; `requirements.txt`
locks their complete dependency set and allowed artifact hashes for **Linux
x86_64 / CPython 3.14**. It includes the pinned build dependency from
`requirements-dev.in`. Do not reuse this platform lock as a claim of Windows,
macOS, ARM, or GPU support. Optuna is intentionally deferred until V5.

| Area | Baseline |
| --- | --- |
| Training runtime | PyTorch 2.14.1, Torchvision 0.29.1, Ultralytics 8.4.174 |
| Generator | NumPy 2.5.3, OpenCV 5.0.0.93, Pillow 12.3.0, Matplotlib 3.11.2, Seaborn 0.13.2 |
| API contracts/runtime | FastAPI 0.142.2, Pydantic 2.13.5, Uvicorn 0.54.0 |
| Existing tracking/data tools | MLflow 3.17.0, DVC 3.67.1 |
| Frontend | React/React DOM 19.2.8, TypeScript 5.9.3, Vite 7.3.1 |
| Build/resolution | setuptools 84.0.0, uv 0.8.22 for lock maintenance |

The Ultralytics, NumPy, OpenCV, and Pillow choices retain the issue 3–7 report's
runtime versions where available. PyTorch/Torchvision compatibility is checked
with an actual CPU detection operation. Frontend dependencies and integrity
hashes are in `frontend/package-lock.json`; Rollup is explicitly held at 4.60.4
for a verified native dependency pair. Vite's Node requirements are documented
in its [getting-started guide](https://vite.dev/guide/).

## Create the Python environment

From the repository root, with Python 3.14 installed:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install setuptools==84.0.0
.venv/bin/python -m pip install --require-hashes --no-build-isolation -r requirements.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pip check
```

These commands create an isolated environment and install dependencies and the
editable `foxtrot-local` package. Editable means source changes are immediately
visible to that environment. They do not run generation or training. The first
installation needs network access and enough disk for the ML runtime. Do not
install into system Python or replace an environment used by active training.

The lock includes both the cached ANTLR wheel hash and its original source archive
hash, so installation can build that dependency from its public source release.
The build dependency is installed first because build isolation is disabled.

For the existing CLI, activate the environment with `source .venv/bin/activate`.
The README's direct scripts and `dvc.yaml` stage commands remain unchanged.
Do not run `dvc repro` merely to check setup: it can generate data, train, or
reuse cached stage outputs. Dataset YAML, glyph caches, and output isolation are
addressed by later milestones, not by this package skeleton.

## Focused verification

Run from the repository root:

```bash
.venv/bin/python -B scripts/verify_baseline.py
.venv/bin/python -m pip check
```

The first command imports every installed skeleton package from a temporary
working directory, checks imports do not load training or credentials, runs the
repository suite (including the 21 issue 3–7 regressions), and compares protected
file hashes before and after. Tests write their datasets and caches in temporary
fixtures. A nonzero
exit code means a check failed or protected content changed. No training or
MLflow upload is started. Hashing also detects newly created protected files;
Python bytecode caches are excluded. The second command checks dependency
compatibility. Neither establishes model quality.

Frontend commands, from `frontend/`:

```bash
npm ci --ignore-scripts --no-audit --no-fund
npm run check
```

`npm ci` installs the locked development dependencies. `check` type-checks and
builds a tiny disposable React fixture under the system temporary directory and
removes it afterward. It does not implement a product screen or start a server.
The actual browser shell belongs to H7. Members will receive bundled assets and
will not need Node once application packaging is delivered.

## Package boundaries

| Path | Responsibility when implemented | Dependency direction |
| --- | --- | --- |
| `src/foxtrot/config/` | H2 request/result contracts and validation | Independent of API, tracking, and execution |
| `src/foxtrot/persistence/` | H3/H4 workspaces, SQLite migrations, durable state/events | Uses configuration; no training imports |
| `src/foxtrot/execution/` | H5 worker and owned subprocess control | Uses contracts and persistence; never browser-owned |
| `src/foxtrot/pipeline/` | H6 explicit-input pipeline adapters | Training libraries load only in execution paths |
| `src/foxtrot/tracking/` | H6 explicit run associations and upload operations | Uses contracts/persistence; credentials stay local |
| `src/foxtrot/api/` | H7 local HTTP interface and bundled static assets | Submits work; does not perform heavy execution |
| `frontend/` | H7 browser shell and subsequent workflow screens | Talks to the API and its authoritative schemas |

At H1 these packages contained only boundary descriptions. H2 implemented the
configuration contracts, H3 implemented local storage and queue claims, and H4
added durable states/events and ownership reconciliation; see
[persistence.md](persistence.md) and [lifecycle.md](lifecycle.md).
H5 implements fixture worker/process supervision; see [execution.md](execution.md).
Real pipeline/tracking adapters and API/browser integration remain later work.
Setuptools
explicitly packages `foxtrot*` under `src/`; it does not accidentally distribute
legacy scripts, datasets, generator resources, tests, or credentials. See
[setuptools package discovery](https://setuptools.pypa.io/en/stable/userguide/package_discovery.html).

Keep `src/class_names.py` lightweight and its class order unchanged. Keep current
scripts and DVC entry points as callers while adapters are introduced in H6 and
V2. This baseline does not refactor the generator or training code.

## Maintaining locks

Only regenerate a lock as an intentional dependency change on the verified target,
then rerun the relevant checks. With uv 0.8.22 available, write the existing
`name==version` lines to `<current-pins.txt>` and pass them as constraints so
versions stay fixed unless a change is intended. Write to a fresh output file:
uv reuses hashes from an existing output file, which keeps stale hashes when the
Python version changes. Restore the three-line header comment afterwards:

```bash
uv pip compile pyproject.toml requirements-dev.in -c <current-pins.txt> --python-version 3.14 --python-platform x86_64-unknown-linux-gnu --generate-hashes --no-header --no-annotate -o requirements.txt
```

During H1, resolution and hash-checked installation used existing cached public
package wheels to avoid repeated large downloads. Cached artifacts were resolved
offline; public PyPI metadata was also inspected for current versions. A clean
second environment installed the locked set and the built package wheel. This is
same-machine reproducibility evidence, not a second-machine or entirely uncached
network-installation check. Future regeneration against PyPI can include more
platform artifact hashes; review those changes rather than silently replacing the
verified lock. Never commit cache directories or machine-local artifact paths.
