import asyncio
import logging
import random
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Iterable, Optional

import httpx


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int
    base_delay_s: float
    max_delay_s: float
    retryable_status_codes: frozenset[int]


def retry_after_seconds(headers: httpx.Headers) -> Optional[float]:
    value = headers.get("retry-after")
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
        if dt is None:
            return None
        return max(0.0, (dt - dt.now(tz=dt.tzinfo)).total_seconds())
    except Exception:
        return None


def _compute_backoff_s(attempt: int, base_delay_s: float, max_delay_s: float) -> float:
    delay = min(max_delay_s, base_delay_s * (2 ** max(0, attempt - 1)))
    # Full jitter (AWS style): random between 0 and computed delay
    return random.random() * delay


async def sleep_before_retry(
    *,
    attempt: int,
    config: RetryConfig,
    retry_after_s: Optional[float],
    reason: str,
) -> None:
    delay_s = _compute_backoff_s(attempt, config.base_delay_s, config.max_delay_s)
    if retry_after_s is not None:
        delay_s = max(delay_s, min(config.max_delay_s, retry_after_s))
    logging.warning(f"Upstream retrying (attempt {attempt}/{config.max_attempts}) in {delay_s:.2f}s: {reason}")
    await asyncio.sleep(delay_s)


async def post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict,
    content: str,
    timeout: httpx.Timeout,
    retry_config: RetryConfig,
    retryable_exceptions: Iterable[type[Exception]] = (
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.WriteError,
        httpx.PoolTimeout,
        httpx.RemoteProtocolError,
    ),
) -> httpx.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retry_config.max_attempts + 1):
        try:
            resp = await client.post(url, headers=headers, content=content, timeout=timeout)
            if resp.status_code in retry_config.retryable_status_codes and attempt < retry_config.max_attempts:
                retry_after_s = retry_after_seconds(resp.headers)
                await resp.aclose()
                await sleep_before_retry(
                    attempt=attempt + 1,
                    config=retry_config,
                    retry_after_s=retry_after_s,
                    reason=f"status={resp.status_code}",
                )
                continue
            return resp
        except tuple(retryable_exceptions) as e:
            last_exc = e
            if attempt >= retry_config.max_attempts:
                raise
            await sleep_before_retry(
                attempt=attempt + 1,
                config=retry_config,
                retry_after_s=None,
                reason=f"{type(e).__name__}: {e}",
            )
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("post_with_retry exhausted without response")
