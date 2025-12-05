"""
Main FastAPI application for Geminicli2api.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .routes.gemini import router as gemini_router
from .routes.openai import router as openai_router
from .services.auth import get_credentials, get_user_project_id, onboard_user
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
    lifespan=lifespan,
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


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": APP_NAME,
        "description": "OpenAI-compatible API proxy for Google's Gemini models",
        "version": APP_VERSION,
        "endpoints": {
            "openai_compatible": {
                "chat_completions": "/v1/chat/completions",
                "models": "/v1/models",
            },
            "native_gemini": {
                "models": "/v1beta/models",
                "generate": "/v1beta/models/{model}:generateContent",
                "stream": "/v1beta/models/{model}:streamGenerateContent",
            },
            "health": "/health",
        },
        "authentication": "Required for all endpoints except root and health",
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": APP_NAME}


app.include_router(openai_router)
app.include_router(gemini_router)
