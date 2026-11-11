# H7 local application shell

One command starts the local website/API and the worker. The browser shows
setup, a training form, and job progress. Training is still the H6 **fixture**
adapter; real YOLO and MLflow arrive in V1.

## Start it

From the repository root, with an absolute workspace outside the checkout:

```bash
.venv/bin/python -B -m foxtrot.api.launcher --workspace /abs/foxtrot-workspace --source-checkout /abs/MlOps --pipeline-adapter fixture
```

It initializes the workspace/database, starts the worker as a separate process,
serves `http://127.0.0.1:8765/`, and opens a browser tab (`--no-browser` to skip).
`--port` changes the port, `--no-worker` serves the website/API only. Omitting
`--pipeline-adapter` runs the H5 infrastructure fixture instead. Ctrl+C stops the
website, then asks the worker to stop and waits for its bounded cleanup; queued
jobs stay queued. Closing the browser never stops anything.

If another worker already holds the lease, the launcher says so and keeps serving;
that worker processes the queue.

## Processes

```text
launcher ──┬── uvicorn (in-process): FastAPI API + built interface, 127.0.0.1 only
           └── worker (separate process, own session) ── supervisor ── child per job
```

The API only records and reads state in SQLite; it never trains.

## Code responsibilities

| Module | Responsibility |
| --- | --- |
| `api.app.create_app` | Routes, security headers, static interface, fallback page when not built |
| `api.security` | Loopback Host check, same-origin + session token for state changes |
| `api.settings` | Tracking destinations; credentials in private `settings/secrets.json`; log redaction |
| `api.launcher` | One command: initialize, start worker, serve, coordinated shutdown |
| `frontend/src` | React shell: Setup, New training, Activity (polling with event cursor) |
| `api/static/` | Vite build output shipped in the Python package |

## API

| Method and path | Purpose |
| --- | --- |
| `GET /api/health` | Liveness and database schema version |
| `GET /api/capabilities` | Version, models, devices, adapters, whether real training exists, destinations |
| `GET /api/schemas` | JSON Schema of the job specification (forms read defaults from it) |
| `GET /api/session` | Per-launch session token (same-origin pages only) |
| `GET /api/datasets` | Registered dataset manifests |
| `GET /api/jobs`, `GET /api/jobs/{id}` | Job list; detail with resolved settings and tracking run link |
| `GET /api/jobs/{id}/events?after=N` | Events after the last seen sequence |
| `POST /api/jobs` | Submit `{submission_key, request}`; repeats return the same job; field errors are 422 |
| `POST /api/jobs/{id}/cancel` | Durable cancellation request |
| `GET/PUT /api/tracking/destinations[/{id}]` | Public destination list; save destination and write-only credentials |

## Security decisions

- Binds to `127.0.0.1` only; there is no option to listen on the network.
- `Host` must be `127.0.0.1` or `localhost` (blocks DNS rebinding).
- POST/PUT/PATCH/DELETE require the `X-Foxtrot-Session` token, and any `Origin`
  must be this app. No CORS headers are sent, so other sites cannot read the token.
- CSP `default-src 'self'; frame-ancestors 'none'`, `nosniff`, `no-referrer`,
  `no-store` on API responses. Interactive API docs are disabled.
- Credentials are write-only: the browser sees only `has_credentials`. They live
  in `settings/secrets.json` (0600), never in job specs, `tracking.json`, events,
  or exports. URLs with embedded `user:pass@` are rejected. A logging filter
  replaces known credential values with `***`.
- Scope: one OS user on one machine. Other local accounts are out of scope.

## Frontend development

```bash
cd frontend && npm run build    # type-check, then build into src/foxtrot/api/static
cd frontend && npm run dev      # Vite dev server; proxies /api to a running launcher on 8765
```

Team members do not need Node: the built files ship with the package.

## Limits

- Progress uses 1-second polling with the event cursor; Server-Sent Events can come with V1.
- Registering datasets, the real trainer, MLflow uploads from the UI, and the
  history/datasets screens arrive with their slices (V1+).
- Tracking destinations are stored but not yet used by an uploader (V1).
