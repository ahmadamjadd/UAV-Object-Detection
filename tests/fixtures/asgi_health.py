"""Bounded in-process HTTP/ASGI check; no socket server or product API."""

import asyncio
import json
import sys
import time

from fastapi import FastAPI

from foxtrot.persistence.jobs import get_job_record


workspace, checkout, job_id = sys.argv[1:]
app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok", "execution_state": get_job_record(workspace, checkout, job_id).execution_state}


async def main():
    started = time.monotonic()
    for _ in range(10):
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "method": "GET", "path": "/health",
                 "root_path": "", "query_string": b"", "headers": [], "scheme": "http", "http_version": "1.1",
                 "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 8765)}
        await asyncio.wait_for(app(scope, receive, send), timeout=2)
        assert messages[0]["status"] == 200
        body = b"".join(message.get("body", b"") for message in messages[1:])
        assert json.loads(body) == {"status": "ok", "execution_state": "training"}
    print(json.dumps({"requests": 10, "seconds": time.monotonic() - started}))


asyncio.run(main())
