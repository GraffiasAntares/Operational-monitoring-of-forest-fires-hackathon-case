"""Small in-process async TTL cache with request coalescing for MVP workloads."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class AsyncTtlCache:
    def __init__(self) -> None:
        self._values: dict[object, tuple[float, object]] = {}
        self._locks: dict[object, asyncio.Lock] = {}

    async def get_or_set(self, key: object, ttl_seconds: float, factory: Callable[[], Awaitable[T]]) -> T:
        now = time.monotonic()
        cached = self._values.get(key)
        if cached and cached[0] > now:
            return cached[1]  # type: ignore[return-value]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._values.get(key)
            if cached and cached[0] > time.monotonic():
                return cached[1]  # type: ignore[return-value]
            value = await factory()
            self._values[key] = (time.monotonic() + ttl_seconds, value)
            return value
