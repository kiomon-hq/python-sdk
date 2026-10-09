"""Shared test doubles: a recording transport and small request helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pytest

from kiomon import Kiomon
from kiomon._transport import Request, Response

Handler = Callable[[Request], Any]

__all__ = [
    "FakeAsyncTransport",
    "FakeTransport",
    "header_of",
    "json_body",
    "json_response",
    "make_client",
    "path_of",
    "query_of",
]


class FakeTransport:
    """Transport double: records every request and returns canned responses.

    The handler may return a :class:`Response`, or raise/return an exception to
    simulate transport failure.
    """

    def __init__(self, handler: Handler | None = None) -> None:
        self.handler: Handler = handler if handler is not None else (lambda request: json_response({"ok": True}))
        self.requests: list[Request] = []

    def send(self, request: Request) -> Response:
        self.requests.append(request)
        result = self.handler(request)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeAsyncTransport:
    """Async transport double, mirroring :class:`FakeTransport`."""

    def __init__(self, handler: Handler | None = None) -> None:
        self.handler: Handler = handler if handler is not None else (lambda request: json_response({"ok": True}))
        self.requests: list[Request] = []
        self.closed = False

    async def send(self, request: Request) -> Response:
        self.requests.append(request)
        result = self.handler(request)
        if isinstance(result, BaseException):
            raise result
        return result

    async def aclose(self) -> None:
        self.closed = True


def json_response(payload: Any, status: int = 200, headers: Mapping[str, str] | None = None) -> Response:
    return Response(
        status=status,
        headers={"content-type": "application/json", **dict(headers or {})},
        body=json.dumps(payload).encode("utf-8"),
    )


def make_client(transport: FakeTransport, **overrides: Any) -> Kiomon:
    options: dict[str, Any] = {
        "api_key": "sk_kiomon_test",
        "base_url": "https://api.example.test",
        "transport": transport,
    }
    options.update(overrides)
    return Kiomon(**options)


def path_of(request: Request) -> str:
    return urlsplit(request.url).path


def query_of(request: Request) -> dict[str, str]:
    return dict(parse_qsl(urlsplit(request.url).query))


def json_body(request: Request) -> Any:
    return json.loads(request.body.decode("utf-8")) if request.body else None


def header_of(request: Request, name: str) -> str | None:
    wanted = name.lower()
    for key, value in request.headers.items():
        if key.lower() == wanted:
            return value
    return None


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace real synchronous backoff waits with a recording no-op."""
    recorded: list[float] = []
    monkeypatch.setattr("kiomon._client._sleep", recorded.append)
    return recorded


@pytest.fixture
def async_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace real asynchronous backoff waits with a recording no-op."""
    recorded: list[float] = []

    async def record(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr("kiomon._async._async_sleep", record)
    return recorded
