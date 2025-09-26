# backend/rate_limit.py
import re
import time
import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Pattern
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.requests import Request

@dataclass(frozen=True)
class Rule:
    pattern: Pattern[str]
    limit: int          # requests
    window: int         # seconds
    scope: str          # 'ip' | 'user' | 'ip-user'

def _compile(pattern: str) -> Pattern[str]:
    return re.compile(pattern)

def _now() -> float:
    return time.time()

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple, in-memory sliding-window rate limiter.
    Keys by IP, by bearer token (best-effort), or their combination.
    Suitable for single-instance deployments (SQLite/uvicorn).
    """
    def __init__(self, app, rules: List[Rule], default_limit: Optional[Rule] = None):
        super().__init__(app)
        self.rules = rules
        self.default = default_limit
        self.bucket: Dict[str, Deque[float]] = {}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        rule = next((r for r in self.rules if r.pattern.search(path)), self.default)
        if rule:
            key = self._key(request, rule.scope, path)
            allowed, retry_after, remaining = self._check_and_push(key, rule.limit, rule.window)
            if not allowed:
                return JSONResponse(
                    {
                        "detail": "Rate limit exceeded. Please retry later.",
                        "retry_after": retry_after,
                    },
                    status_code=429,
                    headers={
                        "Retry-After": str(int(retry_after)),
                        "X-RateLimit-Limit": str(rule.limit),
                        "X-RateLimit-Remaining": str(max(0, remaining)),
                    },
                )

        response = await call_next(request)
        return response

    def _key(self, request: Request, scope: str, path: str) -> str:
        ip = request.client.host if request.client else "unknown"
        auth = request.headers.get("authorization", "")
        token = ""
        if auth.startswith("Bearer "):
            token = auth[7:]
        tokenhash = hashlib.sha1(token.encode("utf-8")).hexdigest()[:16] if token else "anon"

        if scope == "ip":
            return f"ip:{ip}:{path}"
        elif scope == "user":
            return f"user:{tokenhash}:{path}"
        else:  # ip-user
            return f"ipuser:{ip}:{tokenhash}:{path}"

    def _check_and_push(self, key: str, limit: int, window: int):
        now = _now()
        dq = self.bucket.get(key)
        if dq is None:
            dq = deque()
            self.bucket[key] = dq

        # drop old timestamps
        cutoff = now - window
        while dq and dq[0] < cutoff:
            dq.popleft()

        remaining = limit - len(dq) - 1
        if len(dq) >= limit:
            retry_after = max(1, int(dq[0] + window - now))
            return False, retry_after, 0

        dq.append(now)
        return True, 0, remaining

# Helper to construct rules
def rules_for_env(dev: bool) -> List[Rule]:
    r = [
        Rule(_compile(r"^/token$"),                 limit=10, window=60,   scope="ip"),
        Rule(_compile(r"^/register$"),              limit=5,  window=3600, scope="ip"),
        Rule(_compile(r"^/download_(audio|video)"), limit=30, window=600,  scope="ip-user"),
        Rule(_compile(r"^/download_transcript/?$"), limit=60, window=600,  scope="ip-user"),
        Rule(_compile(r"^/batch/submit$"),          limit=10, window=3600, scope="ip-user"),
    ]
    # Optional default (catch-all) for very noisy paths
    default = Rule(_compile(r"^/(?!docs|redoc|openapi\.json).*$"), limit=120, window=60, scope="ip")
    return r + [default]
