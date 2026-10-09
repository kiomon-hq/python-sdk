"""Kiomon SDK — asyncio HTTP transport.

A genuine asynchronous transport with **no third-party dependency**: HTTP/1.1
over :mod:`asyncio` streams, with TLS from the standard library, chunked and
content-length framing, redirect following, and per-attempt timeouts.

Each request owns its connection and asks the server to close it
(``Connection: close``), so there is no pool to size, drain or leak. That keeps
the transport small and predictable; a pooling implementation can be supplied
through the same seam if an application needs one.

Cancellation is real: cancelling the awaiting task cancels the socket work and
closes the connection. Nothing runs on in a background thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import ssl
from collections.abc import Mapping
from typing import Protocol, runtime_checkable
from urllib.parse import SplitResult, urljoin, urlsplit

from ._transport import Request, Response
from .errors import KiomonError

__all__ = ["AsyncioTransport", "AsyncTransport"]

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DEFAULT_PORTS = {"http": 80, "https": 443}
_BODY_METHODS = ("POST", "PUT", "PATCH")


@runtime_checkable
class AsyncTransport(Protocol):
    """The async seam between the client and the network."""

    async def send(self, request: Request) -> Response:
        """Perform one request, or raise :class:`KiomonError` on transport failure."""
        ...  # pragma: no cover - protocol declaration


class AsyncioTransport:
    """Default async transport: standard-library asyncio, no dependencies.

    ``ssl_context`` lets callers pin a CA bundle (or a client certificate) when
    the default context is not what their deployment needs. Unlike
    :class:`~kiomon._transport.UrllibTransport`, this transport does not read
    proxy environment variables; pass a custom transport if you need a proxy.
    """

    def __init__(self, *, ssl_context: ssl.SSLContext | None = None, max_redirects: int = 5) -> None:
        self._ssl_context = ssl_context
        self._default_ssl_context: ssl.SSLContext | None = None
        self._max_redirects = max_redirects

    async def aclose(self) -> None:
        """No persistent resources: every request owns its own connection."""

    async def send(self, request: Request) -> Response:
        current = request
        for _hop in range(self._max_redirects + 1):
            response = await self._send_once(current)
            redirected = _redirect_for(current, response)
            if redirected is None:
                return response
            current = redirected
        raise KiomonError(
            f"Too many redirects (more than {self._max_redirects}).",
            code="network_error",
        )

    # ── One attempt ────────────────────────────────────────────────────────────

    async def _send_once(self, request: Request) -> Response:
        try:
            # wait_for bounds the whole exchange — connect, write and read —
            # and cancels it on expiry, which closes the connection in the
            # exchange's ``finally``.
            return await asyncio.wait_for(self._exchange(request), request.timeout)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as exc:
            raise KiomonError(
                f"Kiomon timed out after {request.timeout:g}s.",
                code="timeout",
                status=504,
                cause=exc,
            ) from exc
        except KiomonError:
            raise
        except UnicodeError as exc:
            raise KiomonError(
                f"Request headers must be latin-1 encodable: {exc}",
                code="invalid_request",
                cause=exc,
            ) from exc
        except (OSError, EOFError, asyncio.LimitOverrunError) as exc:
            raise KiomonError(
                f"Could not reach {_origin(request.url)}: {exc}",
                code="network_error",
                cause=exc,
            ) from exc

    async def _exchange(self, request: Request) -> Response:
        parts = urlsplit(request.url)
        scheme = parts.scheme.lower()
        host = parts.hostname
        if scheme not in _DEFAULT_PORTS or not host:
            raise KiomonError(f"Invalid URL: {request.url!r}", code="invalid_request")
        try:
            port = parts.port or _DEFAULT_PORTS[scheme]
        except ValueError as exc:
            raise KiomonError(f"Invalid port in URL: {request.url!r}", code="invalid_request") from exc

        tls = scheme == "https"
        reader, writer = await asyncio.open_connection(
            host,
            port,
            ssl=self._context() if tls else None,
            server_hostname=host if tls else None,
        )
        try:
            await self._write_request(writer, parts, request)
            return await self._read_response(reader, request.method)
        finally:
            writer.close()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await writer.wait_closed()

    def _context(self) -> ssl.SSLContext:
        if self._ssl_context is not None:
            return self._ssl_context
        if self._default_ssl_context is None:
            self._default_ssl_context = ssl.create_default_context()
        return self._default_ssl_context

    # ── Writing ────────────────────────────────────────────────────────────────

    async def _write_request(self, writer: asyncio.StreamWriter, parts: SplitResult, request: Request) -> None:
        target = parts.path or "/"
        if parts.query:
            target = f"{target}?{parts.query}"

        lines = [f"{request.method} {target} HTTP/1.1", f"Host: {_host_header(parts)}"]
        supplied = {name.lower() for name in request.headers}
        if "content-length" not in supplied and "transfer-encoding" not in supplied:
            if request.body is not None:
                lines.append(f"Content-Length: {len(request.body)}")
            elif request.method.upper() in _BODY_METHODS:
                lines.append("Content-Length: 0")
        if "connection" not in supplied:
            lines.append("Connection: close")
        for name, value in request.headers.items():
            if name.lower() in ("host", "connection"):
                continue
            lines.append(f"{name}: {value}")

        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        if request.body:
            writer.write(request.body)
        await writer.drain()

    # ── Reading ────────────────────────────────────────────────────────────────

    async def _read_response(self, reader: asyncio.StreamReader, method: str) -> Response:
        status, headers = await _read_head(reader)
        # Skip informational responses (100 Continue and friends).
        while 100 <= status < 200:
            status, headers = await _read_head(reader)
        body = await _read_body(reader, method, status, headers)
        return Response(status=status, headers=headers, body=body)


async def _read_head(reader: asyncio.StreamReader) -> tuple[int, dict[str, str]]:
    raw = await reader.readuntil(b"\r\n\r\n")
    text = raw.decode("latin-1")
    lines = text[:-4].split("\r\n")
    if not lines or not lines[0]:
        raise _malformed("empty status line")

    status_parts = lines[0].split(" ", 2)
    if len(status_parts) < 2 or not status_parts[0].upper().startswith("HTTP/"):
        raise _malformed(f"bad status line {lines[0]!r}")
    try:
        status = int(status_parts[1])
    except ValueError as exc:
        raise _malformed(f"bad status code {status_parts[1]!r}") from exc

    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, separator, value = line.partition(":")
        if not separator:
            raise _malformed(f"bad header line {line!r}")
        key = name.strip().lower()
        value = value.strip()
        headers[key] = f"{headers[key]}, {value}" if key in headers else value
    return status, headers


async def _read_body(
    reader: asyncio.StreamReader,
    method: str,
    status: int,
    headers: Mapping[str, str],
) -> bytes:
    if method.upper() == "HEAD" or status in (204, 304):
        return b""

    if "chunked" in headers.get("transfer-encoding", "").lower():
        return await _read_chunked(reader)

    content_length = headers.get("content-length")
    if content_length is not None:
        try:
            length = int(content_length)
        except ValueError as exc:
            raise _malformed(f"bad content-length {content_length!r}") from exc
        if length < 0:
            raise _malformed(f"negative content-length {content_length!r}")
        return await reader.readexactly(length)

    # No framing: the server closes the connection (we asked it to).
    return await reader.read()


async def _read_chunked(reader: asyncio.StreamReader) -> bytes:
    chunks: list[bytes] = []
    while True:
        line = await reader.readuntil(b"\r\n")
        size_text = line[:-2].split(b";", 1)[0].strip()
        try:
            size = int(size_text, 16)
        except ValueError as exc:
            raise _malformed(f"bad chunk size {size_text!r}") from exc
        if size == 0:
            while True:  # consume trailers
                trailer = await reader.readuntil(b"\r\n")
                if trailer == b"\r\n":
                    break
            return b"".join(chunks)
        chunks.append(await reader.readexactly(size))
        await reader.readexactly(2)  # trailing CRLF


# ── Redirects ────────────────────────────────────────────────────────────────


def _redirect_for(request: Request, response: Response) -> Request | None:
    """Build the next request for a redirect, or ``None`` if this is the end.

    Follows fetch's method rules: 301/302 turn a POST into a GET, 303 turns
    everything but HEAD into a GET, and 307/308 preserve method and body.
    Credentials are dropped when the target origin differs, so a redirect can
    never leak an API key to another host.
    """
    if response.status not in _REDIRECT_STATUSES:
        return None
    location = response.headers.get("location")
    if not location:
        return None

    target = urljoin(request.url, location)
    method = request.method
    body = request.body
    dropped_body = False
    if response.status == 303 and method.upper() != "HEAD" or response.status in (301, 302) and method.upper() == "POST":
        method, body, dropped_body = "GET", None, True

    headers = request.headers
    if dropped_body:
        headers = {name: value for name, value in headers.items() if name.lower() != "content-type"}
    if _origin(target) != _origin(request.url):
        headers = {
            name: value for name, value in headers.items() if name.lower() not in ("authorization", "x-api-key")
        }
    return Request(method=method, url=target, headers=headers, body=body, timeout=request.timeout)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _malformed(detail: str) -> KiomonError:
    return KiomonError(
        f"Kiomon returned a malformed HTTP response: {detail}",
        code="network_error",
    )


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


def _host_header(parts: SplitResult) -> str:
    host = parts.hostname or ""
    if ":" in host:  # IPv6 literals need brackets in a Host header
        host = f"[{host}]"
    scheme = (parts.scheme or "").lower()
    try:
        port = parts.port
    except ValueError:  # pragma: no cover - validated before we get here
        port = None
    if port in (None, _DEFAULT_PORTS.get(scheme)):
        return host
    return f"{host}:{port}"
