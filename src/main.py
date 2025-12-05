"""
Main FastAPI application for Geminicli2api.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .routes import anthropic_router, gemini_router, openai_router
from .services.auth import (
    get_credentials,
    get_user_project_id,
    initialize_credential_pool,
    onboard_user,
)
from .config import APP_NAME, APP_VERSION, CREDENTIAL_FILE

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _initialize_credentials() -> None:
    """Initialize credentials on startup."""
    # First try to initialize credential pool (multi-credential support)
    pool = initialize_credential_pool()
    if pool and pool.size() > 0:
        logger.info(f"Loaded {pool.size()} credential(s) from pool")
        # Try to onboard with first credential
        try:
            result = pool.get_next()
            if result:
                entry, _ = result
                if entry.credentials:
                    proj_id = get_user_project_id(entry.credentials)
                    if proj_id:
                        onboard_user(entry.credentials, proj_id)
                        logger.info(f"Onboarded with project: {proj_id}")
        except Exception as e:
            logger.warning(f"Onboarding skipped: {e}")
        return

    # Fallback to single credential mode
    env_creds = os.getenv("GEMINI_CREDENTIALS")
    file_exists = os.path.exists(CREDENTIAL_FILE)

    if env_creds or file_exists:
        try:
            creds = get_credentials(allow_oauth_flow=False)
            if creds:
                try:
                    proj_id = get_user_project_id(creds)
                    if proj_id:
                        onboard_user(creds, proj_id)
                        logger.info(f"Onboarded with project: {proj_id}")
                except Exception as e:
                    logger.error(f"Setup failed: {e}")
            else:
                logger.warning("Credentials file exists but could not be loaded")
        except Exception as e:
            logger.error(f"Credential loading error: {e}")
    else:
        logger.info("No credentials found. Starting OAuth flow...")
        try:
            creds = get_credentials(allow_oauth_flow=True)
            if creds:
                try:
                    proj_id = get_user_project_id(creds)
                    if proj_id:
                        onboard_user(creds, proj_id)
                        logger.info(f"Onboarded with project: {proj_id}")
                except Exception as e:
                    logger.error(f"Setup failed: {e}")
            else:
                logger.error("Authentication failed")
        except Exception as e:
            logger.error(f"Authentication error: {e}")


API_DESCRIPTION = """
**Geminicli2api** is a proxy server that provides OpenAI and Anthropic-compatible APIs 
for Google's Gemini models.

## Features

- **OpenAI-compatible** `/v1/chat/completions` endpoint
- **Anthropic-compatible** `/v1/messages` endpoint  
- **Native Gemini** API passthrough
- Multi-credential support with automatic failover
- Streaming responses

## Authentication

All endpoints (except `/health` and `/v1/models`) require authentication via 
`Authorization: Bearer <token>` header. The token should be your Google OAuth access token.

## Model Mapping

| Request Model | Gemini Model |
|--------------|--------------|
| `gpt-4o`, `gpt-4` | `gemini-2.0-flash` |
| `gpt-4o-mini`, `gpt-3.5-turbo` | `gemini-2.0-flash-lite` |
| `o1`, `o1-pro` | `gemini-2.5-pro` |
| `o3`, `o3-mini` | `gemini-2.5-flash` |
| `claude-*` | `gemini-2.5-flash` |
"""

OPENAPI_TAGS = [
    {
        "name": "OpenAI Compatible",
        "description": "OpenAI-compatible endpoints for chat completions and models.",
    },
    {
        "name": "Anthropic Compatible",
        "description": "Anthropic Claude-compatible endpoint for messages.",
    },
    {
        "name": "Gemini Native",
        "description": "Native Google Gemini API passthrough endpoints.",
    },
    {
        "name": "Health",
        "description": "Health check and status endpoints.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Gemini proxy server...")
    _initialize_credentials()
    logger.info("Server started. Authentication required - see .env file")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    redoc_url=None,
    openapi_tags=OPENAPI_TAGS,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "docExpansion": "list",
        "filter": True,
        "tryItOutEnabled": True,
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/{full_path:path}")
async def handle_preflight(request: Request, full_path: str) -> Response:
    """Handle CORS preflight requests."""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        },
    )


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": APP_NAME}


app.include_router(openai_router)
app.include_router(anthropic_router)
app.include_router(gemini_router)
