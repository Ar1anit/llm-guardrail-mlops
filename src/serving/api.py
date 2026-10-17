"""Guardrail classification service, shadow mode only.

``uv run python -m src.serving.api [key=value ...]`` loads ``configs/serving/default.yaml``,
applies the overrides, and serves:

- ``POST /v1/classify``: score one prompt. The response carries the label, the malicious
  probability, the model release, and ``"action": "log_only"``. Callers must not block on it.
- ``GET /v1/model``: the release and lineage of the loaded model.
- ``GET /healthz`` and ``GET /readyz``: liveness and readiness, without authentication.

Startup fails closed: the model directory (written by ``python -m src.registry.fetch``) must
verify against its manifest and match its ``release.json``, and when ``require_auth`` is true
the bearer token must be set in the environment. Logs carry request ids, lengths, labels,
scores, and latency, never prompt text. Validation errors never echo the request body.
"""

import hmac
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from omegaconf import MISSING, OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from src.evaluation.golden_evaluator import GoldenEvaluator

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/serving/default.yaml")
SUPPORTED_MODES = ("shadow",)
MIN_TOKEN_LENGTH = 32
RELEASE_FILE_NAME = "release.json"
EXPORT_DIRECTORY_NAME = "export"
RELEASE_FIELDS = (
    "model_name",
    "selector",
    "registry_version",
    "release",
    "model_sha256",
    "code_revision",
    "dataset_revision",
)


class ServingConfigError(ValueError):
    """The serving configuration, token, or model directory is unusable."""


@dataclass
class ServingSettings:
    """``configs/serving``: network, model location, mode, input limit, and authentication."""

    host: str = MISSING
    port: int = MISSING
    model_directory: Path = MISSING
    mode: str = MISSING
    max_prompt_characters: int = MISSING
    require_auth: bool = MISSING
    auth_token_env: str = MISSING


def load_settings(path: Path, overrides: Sequence[str] = ()) -> ServingSettings:
    """Read the config, apply ``key=value`` overrides, and validate the result."""
    merged = OmegaConf.merge(
        OmegaConf.structured(ServingSettings),
        OmegaConf.load(path),
        OmegaConf.from_dotlist(list(overrides)),
    )
    settings = OmegaConf.to_object(merged)
    assert isinstance(settings, ServingSettings)
    if settings.mode not in SUPPORTED_MODES:
        raise ServingConfigError(
            f"mode must be one of {list(SUPPORTED_MODES)}; enforcement needs calibrated "
            "thresholds and a fail-open or fail-closed decision first.",
        )
    if not 0 < settings.port < 65536:
        raise ServingConfigError(f"port must be between 1 and 65535, got {settings.port}.")
    if settings.max_prompt_characters < 1:
        raise ServingConfigError("max_prompt_characters must be positive.")
    return settings


def read_token(settings: ServingSettings, environment: dict[str, str]) -> str | None:
    """Return the bearer token, or ``None`` when authentication is explicitly disabled."""
    if not settings.require_auth:
        logger.warning("Authentication is disabled; use this only for local development.")
        return None
    token = environment.get(settings.auth_token_env, "")
    if len(token) < MIN_TOKEN_LENGTH:
        raise ServingConfigError(
            f"Set {settings.auth_token_env} to a random token of at least {MIN_TOKEN_LENGTH} "
            "characters, or set require_auth=false for local development.",
        )
    return token


def load_model(
    directory: Path,
    load_evaluator: Callable[[Path], GoldenEvaluator] = GoldenEvaluator.from_export,
) -> tuple[GoldenEvaluator, dict[str, str]]:
    """Verify ``<directory>/export`` and check it is the model ``release.json`` describes."""
    release_path = directory / RELEASE_FILE_NAME
    try:
        release = json.loads(release_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ServingConfigError(
            f"{release_path} does not exist; run src.registry.fetch."
        ) from error
    if not isinstance(release, dict) or not all(
        isinstance(release.get(key), str) for key in RELEASE_FIELDS
    ):
        raise ServingConfigError(
            f"{release_path} must hold the string fields {list(RELEASE_FIELDS)}."
        )
    evaluator = load_evaluator(directory / EXPORT_DIRECTORY_NAME)
    if evaluator.model_sha256 != release["model_sha256"]:
        raise ServingConfigError(f"The model in {directory} is not the one {release_path} names.")
    return evaluator, {key: release[key] for key in RELEASE_FIELDS}


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class ModelInfo(BaseModel):
    name: str
    release: str
    registry_version: str
    model_sha256: str


class ClassifyResponse(BaseModel):
    request_id: str
    label: Literal["benign", "malicious"]
    score: float
    threshold: float
    mode: Literal["shadow"]
    action: Literal["log_only"]
    model: ModelInfo


def create_app(
    settings: ServingSettings,
    evaluator: GoldenEvaluator,
    release: dict[str, str],
    token: str | None,
) -> FastAPI:
    """Build the app around an already verified model; no I/O happens here."""
    app = FastAPI(
        title="LLM guardrail classifier (shadow mode)",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    inference_lock = threading.Lock()  # Fast tokenizers are not safe across threads.
    model_info = ModelInfo(
        name=release["model_name"],
        release=release["release"],
        registry_version=release["registry_version"],
        model_sha256=release["model_sha256"],
    )

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        if token is None:
            return
        scheme, _, presented = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(presented, token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        # The default handler echoes the submitted input, which here is the user's prompt.
        details = [{"loc": item["loc"], "msg": item["msg"]} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        # The model is verified before the app exists, so a running app is ready.
        return {"status": "ready"}

    @app.get("/v1/model", dependencies=[Depends(authorize)])
    def model() -> dict[str, str]:
        return dict(release)

    @app.post("/v1/classify", dependencies=[Depends(authorize)])
    def classify(body: ClassifyRequest) -> ClassifyResponse:
        request_id = uuid.uuid4().hex
        if len(body.prompt) > settings.max_prompt_characters:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"The prompt exceeds {settings.max_prompt_characters} characters.",
            )
        if not body.prompt.strip():
            raise HTTPException(status_code=422, detail="The prompt is empty.")
        started = time.perf_counter()
        try:
            with inference_lock:
                verdict = evaluator.evaluate(body.prompt)
        except ValueError as error:
            logger.error("request=%s classification failed: %s", request_id, type(error).__name__)
            raise HTTPException(status_code=500, detail="Classification failed.") from error
        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "request=%s chars=%d label=%s score=%.4f latency_ms=%.1f",
            request_id,
            len(body.prompt),
            verdict["label"],
            verdict["score"],
            latency_ms,
        )
        return ClassifyResponse(
            request_id=request_id,
            label="malicious" if verdict["label"] == "malicious" else "benign",
            score=verdict["score"],
            threshold=evaluator.threshold,
            mode="shadow",
            action="log_only",
            model=model_info,
        )

    return app


def main(argv: Sequence[str] | None = None) -> int:
    overrides = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    for name in ("src", __name__):
        logging.getLogger(name).setLevel(logging.INFO)
    try:
        settings = load_settings(DEFAULT_CONFIG_PATH, overrides)
        token = read_token(settings, dict(os.environ))
        evaluator, release = load_model(settings.model_directory)
    except ServingConfigError as error:
        logger.error("Refusing to start: %s", error)
        return 2
    logger.info(
        "Serving %s release %s (sha256 %s) in %s mode.",
        release["model_name"],
        release["release"],
        release["model_sha256"],
        settings.mode,
    )
    uvicorn.run(
        create_app(settings, evaluator, release, token),
        host=settings.host,
        port=settings.port,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
