"""The default async transport: real asyncio sockets, framing, redirects, cancellation."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from kiomon import AsyncioTransport, AsyncKiomon, KiomonError
from kiomon._async_transport import _redirect_for
from kiomon._transport import Request, Response


def _http(payload: bytes, status: int = 200, headers: str = "") -> bytes:
    head = f"HTTP/1.1 {status} X\r\n{headers}Content-Length: {len(payload)}\r\n\r\n"
    return head.encode("latin-1") + payload


def _json_http(value: Any, status: int = 200) -> bytes:
    return _http(json.dumps(value).encode("utf-8"), status, "Content-Type: application/json\r\n")


def _chunked_http(value: Any) -> bytes:
    data = json.dumps(value).encode("utf-8")
    parts = [b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: application/json\r\n\r\n"]
    for index in range(0, len(data), 4):
        piece = data[index : index + 4]
        parts.append(f"{len(piece):x}\r\n".encode("ascii") + piece + b"\r\n")
    parts.append(b"0\r\n\r\n")
    return b"".join(parts)


class Harness:
    """A minimal raw HTTP/1.1 server, driven per test by ``responder``."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.disconnected = False
        self.responder: Callable[[dict[str, Any]], Any] | None = None

    async def default_responder(self, request: dict[str, Any]) -> bytes:
        return _json_http({"results": []})

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1")[:-4].split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers: dict[str, str] = {}
            for line in lines[1:]:
                name, separator, value = line.partition(":")
                if separator:
                    headers[name.strip().lower()] = value.strip()
            length = int(headers.get("content-length") or 0)
            body = await reader.readexactly(length) if length else b""
            request = {
                "method": method,
                "target": target,
                "path": target.split("?")[0],
                "headers": headers,
                "body": body,
            }
            self.requests.append(request)

            responder = self.responder or self.default_responder
            payload = responder(request)
            if inspect.isawaitable(payload):
                payload = await payload
            if payload is None:
                await reader.read()  # wait for the client to hang up
                self.disconnected = True
            else:
                writer.write(payload)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            self.disconnected = True
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


@contextlib.asynccontextmanager
async def serve(harness: Harness) -> AsyncIterator[str]:
    server = await asyncio.start_server(harness.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.close()
        await server.wait_closed()


async def _until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition was not met in time")
        await asyncio.sleep(0.005)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def _client(url: str, **overrides: Any) -> AsyncKiomon:
    options: dict[str, Any] = {"api_key": "sk_kiomon_test", "base_url": url, "workspace_id": "ws_1"}
    options.update(overrides)
    return AsyncKiomon(**options)


# ── Request framing ──────────────────────────────────────────────────────────


def test_get_request_round_trip() -> None:
    async def scenario() -> None:
        harness = Harness()
        async with serve(harness) as url:
            result = await _client(url).search_memory(query="vector db", kind="reference")
            assert result == {"results": []}

        request = harness.requests[0]
        assert request["method"] == "GET"
        assert request["target"] == "/api/search?q=vector+db&limit=10&workspace_id=ws_1&kind=reference"
        assert request["headers"]["x-api-key"] == "sk_kiomon_test"
        assert request["headers"]["accept"] == "application/json"
        assert request["headers"]["connection"] == "close"
        assert request["headers"]["host"].startswith("127.0.0.1")

    run(scenario())


def test_post_body_framing_and_idempotency() -> None:
    async def scenario() -> None:
        harness = Harness()
        harness.responder = lambda request: _json_http(
            {"drafted": 1, "promoted": [], "pending": [], "memories": []}
        )
        learnings = [{"kind": "semantic", "title": "T", "body": "B"}]
        async with serve(harness) as url:
            await _client(url).reflect_session(learnings=learnings)

        request = harness.requests[0]
        assert request["method"] == "POST"
        assert request["headers"]["content-type"] == "application/json"
        assert request["headers"]["content-length"] == str(len(request["body"]))
        assert request["headers"]["idempotency-key"]
        assert json.loads(request["body"]) == {"workspace_id": "ws_1", "learnings": learnings}

    run(scenario())


def test_bodyless_post_sends_zero_content_length() -> None:
    async def scenario() -> None:
        harness = Harness()
        harness.responder = lambda request: _json_http({"ok": True})
        async with serve(harness) as url:
            await _client(url).manage_memory(id="m1", action="archive")

        request = harness.requests[0]
        assert request["method"] == "POST"
        assert request["path"] == "/api/documents/m1/archive"
        assert request["headers"]["content-length"] == "0"
        assert request["body"] == b""

    run(scenario())


# ── Response framing ─────────────────────────────────────────────────────────


def test_chunked_response_is_dechunked() -> None:
    async def scenario() -> None:
        harness = Harness()
        harness.responder = lambda request: _chunked_http({"chunked": True, "ok": "yes"})
        async with serve(harness) as url:
            assert await _client(url).request("/api/thing") == {"chunked": True, "ok": "yes"}

    run(scenario())


def test_response_without_framing_reads_to_eof() -> None:
    async def scenario() -> None:
        harness = Harness()
        harness.responder = lambda request: b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n" + b'{"eof": true}'
        async with serve(harness) as url:
            assert await _client(url).request("/api/thing") == {"eof": True}

    run(scenario())


def test_malformed_response_is_a_network_error() -> None:
    async def scenario() -> None:
        harness = Harness()
        harness.responder = lambda request: b"TOTALLY NOT HTTP\r\n\r\n"
        async with serve(harness) as url:
            with pytest.raises(KiomonError) as excinfo:
                await _client(url, max_retries=0).list_workspaces()
        assert excinfo.value.code == "network_error"

    run(scenario())


# ── Redirects ────────────────────────────────────────────────────────────────


def test_redirect_is_followed() -> None:
    async def scenario() -> None:
        harness = Harness()

        def responder(request: dict[str, Any]) -> bytes:
            if request["path"] == "/api/redirect":
                return b"HTTP/1.1 302 Found\r\nLocation: /api/search\r\nContent-Length: 0\r\n\r\n"
            return _json_http({"results": []})

        harness.responder = responder
        async with serve(harness) as url:
            assert await _client(url).request("/api/redirect") == {"results": []}

        assert [request["path"] for request in harness.requests] == ["/api/redirect", "/api/search"]

    run(scenario())


def test_redirect_converts_post_to_get_and_drops_credentials_cross_origin() -> None:
    request = Request(
        method="POST",
        url="https://api.kiomon.com/api/documents",
        headers={"X-API-Key": "secret", "Content-Type": "application/json", "Idempotency-Key": "k"},
        body=b"{}",
        timeout=5.0,
    )
    response = Response(status=302, headers={"location": "https://evil.example/collect"}, body=b"")

    redirected = _redirect_for(request, response)
    assert redirected is not None
    assert redirected.method == "GET"
    assert redirected.body is None
    assert "X-API-Key" not in redirected.headers
    assert "Content-Type" not in redirected.headers


def test_307_preserves_method_body_and_credentials_same_origin() -> None:
    request = Request(
        method="POST",
        url="https://api.kiomon.com/api/documents",
        headers={"X-API-Key": "secret", "Content-Type": "application/json"},
        body=b'{"a": 1}',
        timeout=5.0,
    )
    response = Response(status=307, headers={"location": "/api/documents/alt"}, body=b"")

    redirected = _redirect_for(request, response)
    assert redirected is not None
    assert redirected.method == "POST"
    assert redirected.body == b'{"a": 1}'
    assert redirected.headers["X-API-Key"] == "secret"
    assert redirected.url == "https://api.kiomon.com/api/documents/alt"


def test_redirect_loop_is_bounded() -> None:
    async def scenario() -> None:
        harness = Harness()
        harness.responder = lambda request: b"HTTP/1.1 302 Found\r\nLocation: /api/loop\r\nContent-Length: 0\r\n\r\n"
        transport = AsyncioTransport(max_redirects=2)
        async with serve(harness) as url:
            with pytest.raises(KiomonError) as excinfo:
                await transport.send(
                    Request(method="GET", url=f"{url}/api/loop", headers={}, body=None, timeout=5.0)
                )
        assert excinfo.value.code == "network_error"
        assert "redirects" in str(excinfo.value)
        assert len(harness.requests) == 3

    run(scenario())


# ── Errors, timeouts, cancellation ───────────────────────────────────────────


def test_http_error_status_maps_to_a_kiomon_error() -> None:
    async def scenario() -> None:
        harness = Harness()
        harness.responder = lambda request: _json_http(
            {"error": "Free plan limit reached", "needsUpgrade": True}, status=402
        )
        async with serve(harness) as url:
            with pytest.raises(KiomonError) as excinfo:
                await _client(url).search_memory(query="x")

        assert excinfo.value.code == "quota_exceeded"
        assert excinfo.value.needs_upgrade is True
        assert len(harness.requests) == 1  # an upgrade prompt is not retried

    run(scenario())


def test_timeout_is_reported_as_timeout() -> None:
    async def scenario() -> None:
        harness = Harness()

        async def slow(request: dict[str, Any]) -> bytes:
            await asyncio.sleep(0.5)
            return _json_http({})

        harness.responder = slow
        async with serve(harness) as url:
            with pytest.raises(KiomonError) as excinfo:
                await _client(url, timeout=0.05, max_retries=0).list_workspaces()

        assert excinfo.value.code == "timeout"
        assert excinfo.value.status == 504

    run(scenario())


def test_cancellation_propagates_and_closes_the_connection() -> None:
    async def scenario() -> None:
        harness = Harness()
        harness.responder = lambda request: None  # never answer
        async with serve(harness) as url:
            task = asyncio.create_task(_client(url, timeout=30.0).request("/api/hang"))
            await _until(lambda: bool(harness.requests))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await _until(lambda: harness.disconnected)

    run(scenario())


def test_connection_failure_is_a_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_open_connection(host: str, port: int, **kwargs: Any) -> Any:
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr("kiomon._async_transport.asyncio.open_connection", fake_open_connection)

    request = Request(method="GET", url="http://127.0.0.1:9/x", headers={}, body=None, timeout=1.0)
    with pytest.raises(KiomonError) as excinfo:
        run(AsyncioTransport().send(request))
    assert excinfo.value.code == "network_error"


def test_invalid_url_scheme_is_an_invalid_request() -> None:
    request = Request(method="GET", url="ftp://example.test/x", headers={}, body=None, timeout=1.0)
    with pytest.raises(KiomonError) as excinfo:
        run(AsyncioTransport().send(request))
    assert excinfo.value.code == "invalid_request"


def test_https_uses_tls_and_the_default_port(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, Any] = {}

    async def fake_open_connection(host: str, port: int, **kwargs: Any) -> Any:
        recorded.update(host=host, port=port, **kwargs)
        raise OSError("stop here")

    monkeypatch.setattr("kiomon._async_transport.asyncio.open_connection", fake_open_connection)

    request = Request(
        method="GET",
        url="https://api.kiomon.com/api/search",
        headers={},
        body=None,
        timeout=1.0,
    )
    with pytest.raises(KiomonError) as excinfo:
        run(AsyncioTransport().send(request))

    assert excinfo.value.code == "network_error"
    assert recorded["host"] == "api.kiomon.com"
    assert recorded["port"] == 443
    assert recorded["ssl"] is not None
    assert recorded["server_hostname"] == "api.kiomon.com"


def test_aclose_is_a_noop() -> None:
    assert run(AsyncioTransport().aclose()) is None


def test_async_transport_protocol_is_runtime_checkable() -> None:
    from kiomon import AsyncTransport

    assert isinstance(AsyncioTransport(), AsyncTransport)
