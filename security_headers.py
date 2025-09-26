# backend/security_headers.py
from __future__ import annotations
from typing import Optional, Dict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.requests import Request

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add sane security headers. Accepts kwargs so Starlette's add_middleware can pass them.
    Usage in main.py:
        app.add_middleware(
            SecurityHeadersMiddleware,
            csp=None,                 # or a CSP string
            hsts=False,               # True only behind HTTPS
            hsts_max_age=31536000,
            hsts_preload=False,
            frame_ancestors="SAMEORIGIN",
            referrer_policy="no-referrer",
            permissions_policy=None,  # e.g. "geolocation=(), microphone=()"
            extra=None,               # dict of extra headers
        )
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        csp: Optional[str] = None,
        hsts: bool = False,
        hsts_max_age: int = 31536000,
        hsts_preload: bool = False,
        frame_ancestors: str = "SAMEORIGIN",
        referrer_policy: str = "no-referrer",
        permissions_policy: Optional[str] = None,
        extra: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(app)
        self.csp = csp
        self.hsts = hsts
        self.hsts_max_age = int(hsts_max_age or 0)
        self.hsts_preload = bool(hsts_preload)
        self.frame_ancestors = frame_ancestors
        self.referrer_policy = referrer_policy
        self.permissions_policy = permissions_policy
        self.extra = extra or {}

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Always-safe defaults
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", self.referrer_policy or "no-referrer")
        response.headers.setdefault("X-XSS-Protection", "0")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")

        # Optional CSP
        if self.csp:
            response.headers["Content-Security-Policy"] = self.csp

        # Optional Permissions-Policy
        if self.permissions_policy:
            response.headers["Permissions-Policy"] = self.permissions_policy

        # Optional HSTS (HTTPS only)
        if self.hsts:
            # Honor common proxy header to detect HTTPS behind a load balancer
            scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
            if (scheme or "").lower() == "https":
                hsts_value = f"max-age={self.hsts_max_age}"
                if self.hsts_preload:
                    hsts_value += "; preload"
                # includeSubDomains is common; omit if you don’t need it
                hsts_value += "; includeSubDomains"
                response.headers["Strict-Transport-Security"] = hsts_value

        # Any extra custom headers
        for k, v in self.extra.items():
            if v is not None:
                response.headers[k] = v

        return response
