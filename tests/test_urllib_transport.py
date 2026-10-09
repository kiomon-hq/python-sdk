"""The default stdlib transport, exercised against real sockets and error objects."""

from __future__ import annotations

import io
import json
import threading
from collections.abc import Iterator
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlsplit

import pytest

from kiomon import Kiomon, KiomonError, UrllibTransport
from kiomon._transport import Request


def _http_request(url: str = "https://api.example.test/api/anything") -> Request:
    return Request(method="GET", url=url, headers={"Accept": "application/json"}, body=None, timeout=5.0)


class _Opener:
    """Stand-in for urllib's opener, raising or returning a canned outcome."""

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome

    def open(self, request: Any, timeout: float | None = None) -> Any:
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _FakeResponse:
    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self.headers = _Headers(headers)
        self._body = body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


class _Headers:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def items(self) -> list[tuple[str, str]]:
        return list(self._values.items())


def test_success_lowercases_headers_and_returns_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = UrllibTransport()
    monkeypatch.setattr(transport, "_opener", _Opener(_FakeResponse(200, {"X-Request-Id": "req_1"}, b"{}")))

    response = transport.send(_http_request())
    assert response.status == 200
    assert response.headers == {"x-request-id": "req_1"}
    assert response.body == b"{}"


def test_http_errors_are_returned_as_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = UrllibTransport()
    headers = Message()
    headers["Content-Type"] = "application/json"
    headers["X-Request-Id"] = "req_1"
    error = HTTPError(
        "https://api.example.test/nope", 404, "Not Found", headers, io.BytesIO(b'{"error": "nope"}')
    )
    monkeypatch.setattr(transport, "_opener", _Opener(error))

    response = transport.send(_http_request("https://api.example.test/nope"))
    assert response.status == 404
    assert response.body == b'{"error": "nope"}'
    assert response.headers["x-request-id"] == "req_1"


def test_a_socket_timeout_becomes_a_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = UrllibTransport()
    monkeypatch.setattr(transport, "_opener", _Opener(URLError(TimeoutError("timed out"))))

    with pytest.raises(KiomonError) as excinfo:
        transport.send(_http_request())
    assert excinfo.value.code == "timeout"
    assert excinfo.value.status == 504
    assert "5s" in str(excinfo.value)


def test_an_unreachable_host_becomes_a_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = UrllibTransport()
    monkeypatch.setattr(transport, "_opener", _Opener(URLError(OSError("connection refused"))))

    with pytest.raises(KiomonError) as excinfo:
        transport.send(_http_request())
    assert excinfo.value.code == "network_error"
    assert "https://api.example.test" in str(excinfo.value)


# ── End to end over a real loopback socket ───────────────────────────────────


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    routes: dict[tuple[str, str], tuple[int, Any]]
    requests: list[dict[str, Any]]


@pytest.fixture
def server() -> Iterator[_Server]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _handle(self) -> None:
            path = urlsplit(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            self.server.requests.append(  # type: ignore[attr-defined]
                {
                    "method": self.command,
                    "path": path,
                    "query": dict(parse_qsl(urlsplit(self.path).query)),
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                    "body": body,
                }
            )
            status, payload = self.server.routes.get(  # type: ignore[attr-defined]
                (self.command, path), (404, {"error": "Not found", "code": "not_found"})
            )
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        do_GET = _handle
        do_POST = _handle
        do_PATCH = _handle

        def log_message(self, *args: Any) -> None:
            pass

    httpd = _Server(("127.0.0.1", 0), Handler)
    httpd.routes = {}
    httpd.requests = []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _client(server: _Server, **overrides: Any) -> Kiomon:
    options: dict[str, Any] = {
        "api_key": "sk_kiomon_test",
        "base_url": f"http://127.0.0.1:{server.server_address[1]}",
        "workspace_id": "ws_1",
    }
    options.update(overrides)
    return Kiomon(**options)


def test_search_end_to_end(server: _Server) -> None:
    server.routes[("GET", "/api/search")] = (200, {"results": []})

    assert _client(server).search_memory(query="vector db", kind="reference") == {"results": []}

    recorded = server.requests[0]
    assert recorded["method"] == "GET"
    assert recorded["query"] == {"q": "vector db", "kind": "reference", "limit": "10", "workspace_id": "ws_1"}
    assert recorded["headers"]["x-api-key"] == "sk_kiomon_test"
    assert recorded["headers"]["x-agent-id"] == "sdk-python"


def test_write_end_to_end(server: _Server) -> None:
    batch = {"drafted": 1, "promoted": [], "pending": [], "memories": []}
    server.routes[("POST", "/api/memories/batch-draft")] = (200, batch)

    learnings = [{"kind": "semantic", "title": "T", "body": "B"}]
    assert _client(server).reflect_session(learnings=learnings) == batch

    recorded = server.requests[0]
    assert recorded["method"] == "POST"
    assert recorded["headers"]["content-type"] == "application/json"
    assert recorded["headers"]["idempotency-key"]
    assert json.loads(recorded["body"].decode("utf-8")) == {"workspace_id": "ws_1", "learnings": learnings}


def test_error_status_end_to_end(server: _Server) -> None:
    server.routes[("GET", "/api/search")] = (402, {"error": "Free plan limit reached", "needsUpgrade": True})

    with pytest.raises(KiomonError) as excinfo:
        _client(server).search_memory(query="x")

    assert excinfo.value.code == "quota_exceeded"
    assert excinfo.value.needs_upgrade is True
    assert len(server.requests) == 1  # an upgrade prompt is not retried
