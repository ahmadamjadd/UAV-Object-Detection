"""Local-only request protections.

The API can start processes and read local files, so a random website open in
the same browser must not be able to drive it:

* the server binds to 127.0.0.1 only (launcher);
* the Host header must be a loopback name, which blocks DNS rebinding;
* state-changing requests need a same-origin Origin header and the per-launch
  session token. The token is readable only by same-origin pages because no
  CORS headers are ever sent.

Other OS accounts on the same machine are outside this local, single-user scope.
"""

import secrets

from fastapi import HTTPException, Request

LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
SESSION_HEADER = "x-foxtrot-session"
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def allowed_origins(port: int) -> set[str]:
    return {f"http://{host}:{port}" for host in LOOPBACK_HOSTS}


def require_local_session(request: Request) -> None:
    """FastAPI dependency for every state-changing endpoint."""
    if request.method not in UNSAFE_METHODS:
        return
    state = request.app.state
    origin = request.headers.get("origin")
    if origin is not None and origin not in allowed_origins(state.port):
        raise HTTPException(status_code=403, detail="Cross-origin request refused")
    supplied = request.headers.get(SESSION_HEADER, "")
    if not secrets.compare_digest(supplied, state.session_token):
        raise HTTPException(status_code=403, detail="Missing or invalid local session")
