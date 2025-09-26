# backend/security_headers.py
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        csp: Optional[str] = None,
        hsts_max_age: int = 31536000,
        include_subdomains: bool = True,
        preload: bool = False,
        enable_hsts: bool = True,
    ):
        super().__init__(app)
        self.csp = csp
        self.enable_hsts = enable_hsts
        self.hsts_value = (
            f"max-age={hsts_max_age}"
            + ("; includeSubDomains" if include_subdomains else "")
            + ("; preload" if preload else "")
        )

    async def dispatch(self, request, call_next):
        resp: Response = await call_next(request)
        h = resp.headers

        # Safe defaults
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("X-Frame-Options", "DENY")
        h.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        h.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=(), payment=()")
        # modern browsers ignore this but it avoids legacy warnings
        h.setdefault("X-XSS-Protection", "0")

        # HSTS only when https is used (or explicitly forced)
        if self.enable_hsts and request.url.scheme == "https":
            h.setdefault("Strict-Transport-Security", self.hsts_value)

        # Content Security Policy can be tricky in dev; pass None there
        if self.csp:
            h.setdefault("Content-Security-Policy", self.csp)

        return resp
