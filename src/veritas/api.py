"""Single-process local research service. Configure a token before remote use."""

import asyncio
import hmac
import json
import logging
import os
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field

from .core import Veritas
from .schema import Document, Query, Result, Strict

WEB = Path(__file__).parent / "web"
LOG = logging.getLogger("veritas.api")


class Batch(Strict):
    documents: list[Document] = Field(min_length=1, max_length=100)


class Retrieve(Query):
    k: int = Field(10, ge=1, le=100)


class Guard:
    def __init__(self, app, token, production):
        self.app = app
        self.token = token
        self.rates = OrderedDict()
        if production and (not token or len(token) < 24):
            raise ValueError("Set VERITAS_TOKEN to at least 24 characters for production mode")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            return await self.app(scope, receive, send)

        async def reject(code, msg):
            await JSONResponse({"detail": msg}, status_code=code)(scope, receive, send)

        headers = dict(scope["headers"])
        if self.token and not hmac.compare_digest(
            headers.get(b"authorization", b""), ("Bearer " + self.token).encode()
        ):
            return await reject(401, "Bearer token required")
        key = (scope.get("client") or ("local",))[0]
        minute = int(time.monotonic() / 60)
        epoch, count = self.rates.get(key, (minute, 0))
        count = count + 1 if epoch == minute else 1
        self.rates[key] = (minute, count)
        self.rates.move_to_end(key)
        if len(self.rates) > 4096:
            self.rates.popitem(last=False)
        if count > 120:
            return await reject(429, "Request rate exceeded")
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > 1_000_000:
                return await reject(413, "Request body exceeds one megabyte")
            if not message.get("more_body", False):
                break

        async def replay():
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        return await self.app(scope, replay, send)


def create_app(engine: Veritas | None = None, token: str | None = None, production=False):
    if production and (not token or len(token) < 24):
        raise ValueError("Production mode requires a bearer token of at least 24 characters")
    pool = ThreadPoolExecutor(max_workers=1)
    cache: OrderedDict[str, Result] = OrderedDict()
    counts: Counter[str] = Counter()
    busy = False

    @asynccontextmanager
    async def lifespan(app):
        yield
        pool.shutdown(wait=True)
        if engine is None:
            app.state.engine.close()

    app = FastAPI(title="VERITAS Evidence API", version="0.2.0", lifespan=lifespan)
    app.state.engine = engine or Veritas.from_config(os.getenv("VERITAS_CONFIG"))
    app.add_middleware(Guard, token=token, production=production)

    async def execute(fn, *args):
        nonlocal busy
        if busy:
            raise HTTPException(503, "Engine busy; retry later")
        busy = True
        future = asyncio.get_running_loop().run_in_executor(pool, fn, *args)

        def release(f):
            nonlocal busy
            busy = False
            if not f.cancelled():
                f.exception()

        future.add_done_callback(release)
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=90)
        except TimeoutError:
            raise HTTPException(504, "Compute deadline exceeded") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/v1/search", response_model=Result)
    @app.post("/v1/verify", response_model=Result)
    @app.post("/v1/evidence/analyze", response_model=Result)
    async def search(req: Query, request: Request):
        counts["queries"] += 1
        fn = app.state.engine.verify_text if request.url.path == "/v1/verify" else app.state.engine.search
        res = await execute(fn, req.query, req.max_search_steps, req.as_of, req.follow_links)
        counts["abstentions"] += res.abstained
        for claim in res.claims:
            cache[claim.id] = res
            if len(cache) > 64:
                cache.popitem(last=False)
        LOG.info(
            json.dumps({"event": "search", "latency_ms": res.latency_ms, "steps": res.search_budget_used})
        )
        return res

    @app.post("/v1/index/documents")
    async def index(req: Batch):
        await execute(app.state.engine.add, req.documents)
        cache.clear()
        return {"upserted": len(req.documents), "total": len(app.state.engine.docs)}

    @app.post("/v1/retrieve")
    async def retrieve(req: Retrieve):
        return await execute(app.state.engine.retrieve, req.query, req.k)

    @app.get("/v1/evidence/{claim_id}", response_model=Result)
    async def evidence(claim_id: str):
        if claim_id not in cache:
            raise HTTPException(404, "Claim unavailable")
        return cache[claim_id]

    @app.get("/v1/health")
    async def health():
        return {"status": "ok", "documents": len(app.state.engine.docs)}

    @app.get("/v1/metrics")
    async def metrics():
        return dict(counts)

    @app.get("/v1/models")
    async def models():
        e = app.state.engine
        return {
            "retrieval": "multilingual MiniLM/HNSW" if e.neural else "BM25 + character TF-IDF/LSA",
            "inference": "DeBERTa ONNX + rules" if e.neural else "conservative statement rules",
            "scientific_evidence": e.config.scientific_evidence,
            "learned_ranker": bool(e.weights),
            "calibration": e.cal["target"] if e.cal else None,
        }

    @app.get("/")
    async def home():
        return FileResponse(WEB / "index.html")

    app.mount("/assets", StaticFiles(directory=str(WEB)), name="assets")
    return app


def factory():
    return create_app(token=os.getenv("VERITAS_TOKEN"), production=os.getenv("VERITAS_ENV") == "production")
