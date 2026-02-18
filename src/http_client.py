import httpx
from typing import Optional

from .config import (
    UPSTREAM_CONNECT_TIMEOUT_S,
    UPSTREAM_READ_TIMEOUT_S,
    UPSTREAM_MAX_CONNECTIONS,
    UPSTREAM_MAX_KEEPALIVE_CONNECTIONS,
)

_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        limits = httpx.Limits(
            max_connections=UPSTREAM_MAX_CONNECTIONS,
            max_keepalive_connections=UPSTREAM_MAX_KEEPALIVE_CONNECTIONS,
        )
        timeout = httpx.Timeout(timeout=None, connect=UPSTREAM_CONNECT_TIMEOUT_S, read=UPSTREAM_READ_TIMEOUT_S)
        _client = httpx.AsyncClient(limits=limits, timeout=timeout)
    return _client


async def close_http_client() -> None:
    global _client
    if _client is None:
        return
    await _client.aclose()
    _client = None
