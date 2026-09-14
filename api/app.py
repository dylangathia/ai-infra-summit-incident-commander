"""HTTP + WebSocket surface. One service, one URL, nothing to coordinate."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.engine import Engine

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="AI Incident Commander")
engine = Engine(provider_name=os.environ.get("IC_PROVIDER", "echo"))


@app.on_event("startup")
async def _startup():
    asyncio.create_task(engine.run())


@app.get("/api/state")
async def state():
    return JSONResponse(engine.state())


@app.post("/api/inject")
async def inject(key: Optional[str] = None):
    info = engine.inject(key)
    # the response confirms a fault was injected without naming it, so a judge
    # can trigger one blind and compare the agent's answer against the reveal
    return JSONResponse({"injected": True, "reveal_key": info["key"],
                         "reveal_label": info["label"]})


@app.post("/api/reset")
async def reset():
    engine.reset()
    return JSONResponse({"ok": True})


@app.post("/api/agent")
async def agent(enabled: bool = True):
    engine.set_agent(enabled)
    return JSONResponse({"agent_enabled": enabled})


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    try:
        while True:
            await socket.send_json(engine.state())
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
    except Exception:
        return


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")