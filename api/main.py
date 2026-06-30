# api/main.py
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from db import init_db
from storage import ensure_bucket


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    ensure_bucket()
    yield


app = FastAPI(title="Furniture Pipeline API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}
