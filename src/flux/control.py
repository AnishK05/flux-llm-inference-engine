"""Live-page helpers: recent TTFT ring and a token-rate window."""

from __future__ import annotations

import time
from collections import deque


class RecentTtft:
    """p50 TTFT over the last `window_s` seconds."""

    def __init__(self, window_s: float = 60.0) -> None:
        self.window_s = window_s
        self._samples: deque[tuple[float, float]] = deque()

    def add(self, ttft_s: float) -> None:
        if ttft_s <= 0:
            return
        now = time.time()
        self._samples.append((now, ttft_s))
        self._trim(now)

    def p50_ms(self) -> float | None:
        self._trim(time.time())
        if not self._samples:
            return None
        values = sorted(item[1] * 1000.0 for item in self._samples)
        mid = (len(values) - 1) // 2
        return values[mid]

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()


class TokenRate:
    """Instantaneous tok/s from a short sliding window of totals."""

    def __init__(self, window_s: float = 5.0) -> None:
        self.window_s = window_s
        self._marks: deque[tuple[float, int]] = deque()

    def observe(self, total_tokens: int) -> float:
        now = time.time()
        self._marks.append((now, total_tokens))
        cutoff = now - self.window_s
        while len(self._marks) > 1 and self._marks[0][0] < cutoff:
            self._marks.popleft()
        if len(self._marks) < 2:
            return 0.0
        dt = self._marks[-1][0] - self._marks[0][0]
        dtok = self._marks[-1][1] - self._marks[0][1]
        if dt <= 0:
            return 0.0
        return max(0.0, dtok / dt)
