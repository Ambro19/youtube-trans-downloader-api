# # timestamp_patch.py
# # Tiny utilities + middleware to guarantee UTC "Z" timestamps in all JSON responses
# # and helpers to use filesystem mtimes for activity items.

# from __future__ import annotations

# import json
# import os
# import re
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Any, Dict, Iterable, Mapping, MutableMapping, Union

# from starlette.middleware.base import BaseHTTPMiddleware
# from starlette.responses import Response

# # --- UTC helpers -------------------------------------------------------------

# _ISO_NO_TZ = re.compile(
#     r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
#     r"(\.\d{1,6})?$"  # optional fraction
# )
# _ISO_WITH_TZ = re.compile(
#     r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
#     r"(\.\d{1,6})?"
#     r"(Z|[+-]\d{2}:\d{2})$"
# )

# DATETIME_KEYS = {"timestamp", "created_at", "updated_at", "occurred_at", "time", "date"}

# def utc_now() -> datetime:
#     return datetime.now(timezone.utc)

# def iso_utc_z(dt: datetime) -> str:
#     """Return ISO8601 ending with Z. Accepts naive or aware datetimes."""
#     if dt.tzinfo is None:
#         dt = dt.replace(tzinfo=timezone.utc)
#     else:
#         dt = dt.astimezone(timezone.utc)
#     return dt.isoformat().replace("+00:00", "Z")

# def ensure_iso_z(value: Union[str, datetime]) -> str:
#     """Normalize a datetime string or object to ISO with trailing Z."""
#     if isinstance(value, datetime):
#         return iso_utc_z(value)
#     if isinstance(value, str):
#         s = value.strip()
#         if _ISO_WITH_TZ.match(s):
#             return s.replace("+00:00", "Z")
#         if _ISO_NO_TZ.match(s):
#             return f"{s}Z"
#     # Not an ISO-like datetime; return as-is
#     return str(value)

# def file_mtime_iso(path: Union[str, Path]) -> str:
#     """Get file modified time as ISO Z (UTC)."""
#     ts = os.path.getmtime(path)
#     dt = datetime.fromtimestamp(ts, tz=timezone.utc)
#     return iso_utc_z(dt)

# # --- JSON transformer --------------------------------------------------------

# def _transform(obj: Any) -> Any:
#     """Recursively ensure timestamps end with Z; only touches datetime-like values
#     and typical datetime keys to avoid altering arbitrary strings."""
#     if isinstance(obj, Mapping):
#         new: Dict[str, Any] = {}
#         for k, v in obj.items():
#             if isinstance(v, datetime):
#                 new[k] = iso_utc_z(v)
#             elif isinstance(v, str):
#                 # Only force when key is date-like OR string looks ISO
#                 if k in DATETIME_KEYS or _ISO_NO_TZ.match(v) or _ISO_WITH_TZ.match(v):
#                     new[k] = ensure_iso_z(v)
#                 else:
#                     new[k] = v
#             else:
#                 new[k] = _transform(v)
#         return new
#     if isinstance(obj, (list, tuple)):
#         return [ _transform(x) for x in obj ]
#     return obj

# # --- Middleware --------------------------------------------------------------

# class EnsureUtcZMiddleware(BaseHTTPMiddleware):
#     """Post-process JSON responses to enforce ISO8601 'Z' timestamps."""
#     async def dispatch(self, request, call_next):
#         resp: Response = await call_next(request)
#         ctype = (resp.headers.get("content-type") or "").lower()
#         if "application/json" not in ctype:
#             return resp

#         # Pull original body (consume iterator), then rebuild response
#         body_bytes = b""
#         async for chunk in resp.body_iterator:
#             body_bytes += chunk

#         try:
#             data = json.loads(body_bytes.decode("utf-8"))
#         except Exception:
#             # If not valid JSON, return untouched
#             resp.body_iterator = iter([body_bytes])  # reattach
#             return resp

#         normalized = _transform(data)
#         new_body = json.dumps(normalized, ensure_ascii=False).encode("utf-8")

#         # Rebuild Response with same status and headers
#         new_headers = dict(resp.headers)
#         new_headers["content-length"] = str(len(new_body))
#         return Response(
#             content=new_body,
#             status_code=resp.status_code,
#             media_type="application/json",
#             headers=new_headers,
#         )

# # --- Activity helpers --------------------------------------------------------

# def enrich_activity_timestamp_with_fs(item: MutableMapping[str, Any], downloads_dir: Union[str, Path]) -> None:
#     """
#     If an activity references a real file (by 'path' or 'filename'), replace/confirm
#     its timestamp using the file's mtime. Always normalized to ISO Z.
#     """
#     downloads_dir = Path(downloads_dir)
#     candidate: Path | None = None

#     # 1) explicit absolute/relative path
#     p = item.get("path") or item.get("file_path")
#     if isinstance(p, str) and p:
#         candidate = Path(p)
#         if not candidate.is_absolute():
#             candidate = downloads_dir / candidate

#     # 2) filename in the downloads dir
#     if candidate is None:
#         name = item.get("filename")
#         if isinstance(name, str) and name:
#             candidate = downloads_dir / name

#     if candidate and candidate.exists():
#         item["timestamp"] = file_mtime_iso(candidate)  # authoritative
#     else:
#         # Fallback: normalize any present timestamp-ish fields
#         if "timestamp" in item:
#             item["timestamp"] = ensure_iso_z(item["timestamp"])
#         elif "created_at" in item:
#             item["timestamp"] = ensure_iso_z(item["created_at"])
#         else:
#             item["timestamp"] = iso_utc_z(utc_now())
