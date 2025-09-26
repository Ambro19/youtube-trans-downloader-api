# security_headers.py
import os
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.requests import Request
from starlette.responses import Response


def _env(*keys: str, default: str = "development") -> str:
    for k in keys:
        v = os.getenv(k)
        if v:
            return v
    return default


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds common security headers. HSTS is enabled by default in production.
    Optional CSP can be supplied via init args or env vars.

    Env vars:
      - ENVIRONMENT / APP_ENV: "production" enables HSTS by default
      - ENABLE_CSP: "1" to enable CSP, otherwise off
      - CSP_VALUE: CSP header value (string)
      - CSP_REPORT_ONLY: "1" to send CSP in Report-Only mode
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        enable_hsts: Optional[bool] = None,
        csp: Optional[str] = None,
        report_only: Optional[bool] = None,
    ) -> None:
        super().__init__(app)

        env = _env("ENVIRONMENT", "APP_ENV").lower()
        self.enable_hsts = (enable_hsts
                            if enable_hsts is not None
                            else env == "production")

        # CSP from args or env
        if csp is None and os.getenv("ENABLE_CSP", "0") == "1":
            csp = os.getenv("CSP_VALUE", "default-src 'self'; base-uri 'self'; frame-ancestors 'none';")
        self.csp = csp

        if report_only is None:
            report_only = os.getenv("CSP_REPORT_ONLY", "0") == "1"
        self.report_only = report_only

    async def dispatch(self, request: Request, call_next) -> Response:
        resp = await call_next(request)

        # Core safe defaults
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        resp.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        resp.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")

        # HSTS only if enabled and the request is HTTPS (prevents local-dev issues)
        if self.enable_hsts and request.url.scheme == "https":
            resp.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=63072000; includeSubDomains; preload",
            )

        # Optional CSP
        if self.csp:
            header = "Content-Security-Policy-Report-Only" if self.report_only else "Content-Security-Policy"
            resp.headers.setdefault(header, self.csp)

        return resp
