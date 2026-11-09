"""FastAPI app. Run:  uvicorn src.api.main:app --port 8000"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src.api.schemas import AskRequest, AskResponse, ClassifyRequest, ClassifyResponse
from src.api.service import Services
from src.utils.logging_utils import get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    services = getattr(app.state, "services", None)
    if services is None:
        services = Services()
        services.load()
        app.state.services = services
        logger.info("models loaded: %s", services.status()["load_report"])
    yield


app = FastAPI(title="Feedback Intelligence", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Process-Time-ms"] = str(ms)
    logger.info("%s %s -> %s in %s ms", request.method, request.url.path, response.status_code, ms)
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/status")
def status(request: Request) -> dict:
    return request.app.state.services.status()


@app.post("/classify", response_model=ClassifyResponse)
def classify(body: ClassifyRequest, request: Request) -> dict:
    return {"results": request.app.state.services.classify(body.texts)}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> dict:
    return request.app.state.services.ask(body.question)