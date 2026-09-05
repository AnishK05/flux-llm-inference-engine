"""Control-plane Redis: rate limits and job status. Never tensors."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class ControlPlane:
    """In-memory job store, plus Redis INCR rate limits when enabled."""

    def __init__(self) -> None:
        self.enabled = False
        self._redis: Any = None
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def connect(self, url: str) -> None:
        import redis

        client = redis.Redis.from_url(url, decode_responses=True)
        client.ping()
        self._redis = client
        self.enabled = True
        logger.info("redis control plane connected")

    def close(self) -> None:
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:  # noqa: BLE001
                pass
        self._redis = None
        self.enabled = False

    def allow(self, key: str, limit: int, window_s: int = 60) -> bool:
        """Return True if this key is under the per-window limit."""
        if not self.enabled or self._redis is None or limit <= 0:
            return True
        redis_key = f"flux:rl:{key}"
        try:
            n = int(self._redis.incr(redis_key))
            if n == 1:
                self._redis.expire(redis_key, window_s)
            return n <= limit
        except Exception:  # noqa: BLE001
            logger.warning("redis rate limit failed; allowing request", exc_info=True)
            return True

    def put_job(self, job_id: str, payload: dict[str, Any], ttl_s: int = 3600) -> None:
        body = dict(payload)
        body["updated_at"] = time.time()
        with self._lock:
            self._jobs[job_id] = body
        if self.enabled and self._redis is not None:
            try:
                self._redis.setex(f"flux:job:{job_id}", ttl_s, json.dumps(body))
            except Exception:  # noqa: BLE001
                logger.warning("redis job write failed", exc_info=True)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        if self.enabled and self._redis is not None:
            try:
                raw = self._redis.get(f"flux:job:{job_id}")
                if raw:
                    return json.loads(raw)
            except Exception:  # noqa: BLE001
                logger.warning("redis job read failed", exc_info=True)
        with self._lock:
            found = self._jobs.get(job_id)
            return dict(found) if found else None


def attach_control_plane(settings: Any) -> ControlPlane:
    plane = ControlPlane()
    if not getattr(settings, "enable_redis", False):
        return plane
    try:
        plane.connect(settings.redis_url)
    except Exception:  # noqa: BLE001
        logger.warning("redis unavailable; control plane stays in-memory", exc_info=True)
    return plane
