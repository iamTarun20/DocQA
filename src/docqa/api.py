"""Minimal HTTP entry point for the DocQA engine."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from docqa.answer import Answer, DocQA, ProviderError, engine_from_env
from docqa.retrieval import Mode


class AskRequest(BaseModel):
    """A nonblank question and a bounded number of passages to inspect."""

    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    top_k: int = Field(default=5, ge=1, le=20)


def create_app(configured: tuple[DocQA, Mode] | None = None) -> FastAPI:
    """Create the API and build one shared index when the server starts."""
    engine: DocQA
    mode: Mode

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal engine, mode
        engine, mode = configured if configured is not None else engine_from_env()
        yield

    app = FastAPI(
        title="DocQA",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report readiness after the index has loaded."""
        return {"status": "ok"}

    @app.post("/ask")
    def ask(request: AskRequest) -> Answer:
        """Return an answer and citation coordinates."""
        try:
            return engine.ask(request.question, mode, request.top_k)
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app
