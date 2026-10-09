"""Kiomon SDK — errors.

One error type crosses the SDK boundary. The API's own error envelope is a
flat ``{"error": "...", "code": "..."}``, optionally with ``needsUpgrade`` and
``retry_after``, so the SDK derives a stable ``code`` from the HTTP status and
prefers a server-supplied ``code`` when the worker sends one.

The taxonomy below mirrors the worker's ``ApiErrorCode``
(``worker/src/lib/api-error.ts``) exactly, so a response carrying a code and one
that had to be inferred from the status map to the same value. Three codes are
raised by this SDK rather than the wire — ``invalid_request`` for bad arguments
(raised before any request), ``timeout`` and ``network_error`` for transport
failures — and ``unknown`` exists so a future server code never crashes a
caller.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Final, Literal

__all__ = [
    "DEFAULT_CODE_BY_STATUS",
    "KiomonError",
    "KiomonErrorCode",
    "code_for_status",
    "error_from_response",
    "is_kiomon_error",
]

#: Stable, switchable error codes.
KiomonErrorCode = Literal[
    # ── Request shape ─────────────────────────────────────────────────────────
    "bad_request",
    #: Schema validation failed — the shape was wrong, not just the values.
    "validation_failed",
    "payload_too_large",
    # ── Credential / permission ───────────────────────────────────────────────
    "unauthorized",
    #: Account is mid-deletion: not a key problem, and not fixable by re-issuing one.
    "account_pending_deletion",
    "forbidden",
    #: The caller's workspace role is too low. Describes the caller, not the resource.
    "insufficient_role",
    #: An OAuth connected app reached outside its allowlist.
    "connector_scope_denied",
    # ── Absence ───────────────────────────────────────────────────────────────
    #: Deliberately undifferentiated: the API fails closed with 404 so it never
    #: confirms whether a resource exists.
    "not_found",
    # ── State conflicts ───────────────────────────────────────────────────────
    "conflict",
    "duplicate",
    #: Same Idempotency-Key, different request body.
    "idempotency_conflict",
    #: Same Idempotency-Key, first request still running.
    "idempotency_in_progress",
    # ── Limits (both are upgrade prompts, not failures) ───────────────────────
    "quota_exceeded",
    "pro_required",
    "rate_limited",
    # ── Infrastructure ────────────────────────────────────────────────────────
    "service_unavailable",
    "internal_error",
    # ── Raised by this SDK, never by the wire ─────────────────────────────────
    "invalid_request",
    "timeout",
    "network_error",
    # ── Fallback for a code this version does not know ────────────────────────
    "unknown",
]

#: Fallback mapping for a response that carried no ``code``.
#:
#: Kept identical to the worker's ``DEFAULT_CODE_BY_STATUS`` on purpose: the
#: same 400 must resolve to ``bad_request`` whether the worker labelled it or
#: the SDK had to infer it. ``timeout`` is absent deliberately — the SDK raises
#: it for its own timed-out requests, while a 504 from the edge is
#: ``service_unavailable``.
DEFAULT_CODE_BY_STATUS: Final[Mapping[int, str]] = {
    400: "bad_request",
    401: "unauthorized",
    402: "quota_exceeded",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    413: "payload_too_large",
    429: "rate_limited",
    500: "internal_error",
    502: "service_unavailable",
    503: "service_unavailable",
    504: "service_unavailable",
}

#: Codes where retrying the same request could plausibly succeed.
_RETRYABLE_CODES: Final[frozenset] = frozenset(
    {"rate_limited", "internal_error", "service_unavailable", "timeout", "network_error"}
)


def code_for_status(status: int) -> str:
    """Map an HTTP status to a code.

    Exported so tests and callers can reason about the mapping.
    """
    known = DEFAULT_CODE_BY_STATUS.get(status)
    if known is not None:
        return known
    if status >= 500:
        return "service_unavailable"
    if status >= 400:
        return "bad_request"
    return "unknown"


def _header(headers: Mapping[str, str] | None, name: str) -> str | None:
    """Case-insensitive header lookup (transports normally pre-lowercase)."""
    if not headers:
        return None
    direct = headers.get(name)
    if direct is not None:
        return direct
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


class KiomonError(Exception):
    """The one error type raised by the SDK.

    Carries ``status``, ``code``, ``body``, ``needs_upgrade``, ``request_id`` and
    ``retry_after`` alongside the message, so callers branch on stable
    identifiers instead of prose.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int = 0,
        code: str | None = None,
        body: Any | None = None,
        needs_upgrade: bool = False,
        request_id: str | None = None,
        retry_after: float | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        #: HTTP status, or 0 when the request never reached the server.
        self.status = status
        self.code = code if code is not None else code_for_status(status)
        #: Raw decoded response body, when there was one. Useful for debugging,
        #: not for control flow.
        self.body = body
        #: ``True`` on a Free-plan quota rejection (HTTP 402) — the caller
        #: should upgrade, not retry.
        self.needs_upgrade = needs_upgrade
        self.request_id = request_id
        #: Server-advertised wait before a retry, in seconds, when it sent one.
        self.retry_after = retry_after
        if cause is not None:
            self.__cause__ = cause

    @property
    def message(self) -> str:
        """The human-readable message (``str(error)`` returns the same thing)."""
        return str(self.args[0]) if self.args else ""

    @property
    def retryable(self) -> bool:
        """Whether retrying the *same* request could plausibly succeed.

        A retry is only safe for writes when an idempotency key is attached —
        the SDK does that automatically, but a hand-rolled retry should pass one
        too. ``quota_exceeded`` and ``pro_required`` are deliberately absent:
        those need an upgrade, not another attempt.
        """
        return self.code in _RETRYABLE_CODES

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        where = f"HTTP {self.status}" if self.status else "no response"
        return f"KiomonError(code={self.code!r}, status={self.status!r}, message={self.message!r}) [{where}]"


def is_kiomon_error(value: Any) -> bool:
    """Whether ``value`` is a :class:`KiomonError` (a narrowing helper)."""
    return isinstance(value, KiomonError)


def error_from_response(
    *,
    status: int,
    body: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    fallback_message: str = "",
) -> KiomonError:
    """Build a :class:`KiomonError` from a non-2xx response.

    The API answers ``{"error": "..."}``, and the rate limiter adds
    ``needsUpgrade: true`` with HTTP 402 on Free. A 404 is deliberately also
    what a *denied* workspace returns — the worker fails closed rather than
    confirming that a workspace exists — so ``not_found`` can mean "no access"
    rather than "does not exist".
    """
    envelope: Mapping[str, Any] | None = None
    if body:
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, Mapping):
            envelope = parsed

    message = fallback_message or f"Request failed with HTTP {status}"
    server_code: str | None = None
    needs_upgrade = status == 402
    retry_after: float | None = None

    if envelope is not None:
        error_text = envelope.get("error")
        message_text = envelope.get("message")
        if isinstance(error_text, str) and error_text:
            message = error_text
        elif isinstance(message_text, str) and message_text:
            message = message_text

        raw_code = envelope.get("code")
        if isinstance(raw_code, str) and raw_code:
            # A server-supplied code wins over the status mapping; an unknown
            # string passes through so a caller can switch on codes this SDK
            # version has never heard of.
            server_code = raw_code

        if envelope.get("needsUpgrade") is True:
            needs_upgrade = True

        raw_retry_after = envelope.get("retry_after")
        if (
            isinstance(raw_retry_after, (int, float))
            and not isinstance(raw_retry_after, bool)
            and raw_retry_after > 0
        ):
            retry_after = float(raw_retry_after)

    header_retry_after = _header(headers, "retry-after")
    if header_retry_after:
        try:
            parsed_header = float(header_retry_after)
        except ValueError:
            parsed_header = 0.0
        if parsed_header > 0:
            retry_after = max(retry_after or 0.0, parsed_header)

    return KiomonError(
        message,
        status=status,
        code=server_code if server_code is not None else code_for_status(status),
        body=envelope,
        needs_upgrade=needs_upgrade,
        request_id=_header(headers, "x-request-id"),
        retry_after=retry_after,
    )
