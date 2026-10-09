"""Kiomon SDK — HTTP transport.

The default transport uses only the standard library (``urllib``), so the
package has zero runtime dependencies. A transport is anything with a
``send(request) -> Response`` method, which is also the seam tests use to run
the whole client without a socket.

Transports translate *transport* failures (DNS, TLS, connection reset, timeout)
into :class:`~kiomon.errors.KiomonError` with ``code="network_error"`` or
``code="timeout"`` so the client's retry loop sees one error type; HTTP error
statuses are returned as ordinary responses and turned into errors by the
client, which preserves the response body for debugging.
"""

from __future__ import annotations

import http.client
import socket
import ssl
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from .errors import KiomonError

__all__ = ["Request", "Response", "Transport", "UrllibTransport"]


@dataclass(frozen=True)
class Request:
    """One prepared HTTP request."""

    method: str
    url: str
    headers: Mapping[str, str]
    body: bytes | None
    timeout: float


@dataclass(frozen=True)
class Response:
    """One HTTP response, with headers lowercased for predictable lookup."""

    status: int
    headers: Mapping[str, str]
    body: bytes


@runtime_checkable
class Transport(Protocol):
    """The seam between the client and the network."""

    def send(self, request: Request) -> Response:
        """Perform one request, or raise :class:`KiomonError` on transport failure."""
        ...  # pragma: no cover - protocol declaration


class UrllibTransport:
    """Default transport: the standard library, so no HTTP library is required.

    ``ssl_context`` lets callers pin a CA bundle (or a client certificate) when
    the default context is not what their deployment needs. Proxy environment
    variables (``HTTPS_PROXY``, ``NO_PROXY``, …) are honoured automatically by
    ``urllib``'s default opener.
    """

    def __init__(self, *, ssl_context: ssl.SSLContext | None = None) -> None:
        handlers = []
        if ssl_context is not None:
            handlers.append(urllib.request.HTTPSHandler(context=ssl_context))
        self._opener = urllib.request.build_opener(*handlers)

    def send(self, request: Request) -> Response:
        http_request = urllib.request.Request(request.url, data=request.body, method=request.method)
        for name, value in request.headers.items():
            http_request.add_header(name, value)

        try:
            with self._opener.open(http_request, timeout=request.timeout) as response:
                return Response(
                    status=int(response.status),
                    headers=_lower_headers(response.headers),
                    body=response.read(),
                )
        except urllib.error.HTTPError as exc:  # a real response, just not 2xx
            try:
                body = exc.read()
            except Exception:  # pragma: no cover - defensive
                body = b""
            return Response(status=int(exc.code), headers=_lower_headers(exc.headers), body=body)
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise _timeout_error(request, exc) from exc
            raise _network_error(request, exc) from exc
        except TimeoutError as exc:
            raise _timeout_error(request, exc) from exc
        except (http.client.HTTPException, OSError) as exc:
            raise _network_error(request, exc) from exc


def _lower_headers(headers: object) -> Mapping[str, str]:
    items = getattr(headers, "items", None)
    if items is None:  # pragma: no cover - defensive
        return {}
    return {str(key).lower(): str(value) for key, value in items()}


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _timeout_error(request: Request, cause: BaseException) -> KiomonError:
    return KiomonError(
        f"Kiomon timed out after {request.timeout:g}s.",
        code="timeout",
        status=504,
        cause=cause,
    )


def _network_error(request: Request, cause: BaseException) -> KiomonError:
    reason = cause.reason if isinstance(cause, urllib.error.URLError) else cause
    return KiomonError(
        f"Could not reach {_origin(request.url)}: {reason}",
        code="network_error",
        cause=cause,
    )
