"""Async client: native coroutines sharing the sync client's contract."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from conftest import FakeAsyncTransport, FakeTransport, header_of, json_body, json_response, path_of

from kiomon import AsyncKiomon, CallOptions, KiomonError

PACKET = {"summary": "ok", "facts": [], "episodes": [], "warnings": [], "conflicts": [], "retrieved_ids": []}
SEARCH_OK = {"results": []}


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def async_client(transport: Any, **overrides: Any) -> AsyncKiomon:
    options: dict = {"api_key": "sk_kiomon_test", "base_url": "https://api.example.test", "transport": transport}
    options.update(overrides)
    return AsyncKiomon(**options)


def test_retrieve_memory_awaits_the_same_request() -> None:
    transport = FakeAsyncTransport(lambda request: json_response(PACKET))
    client = async_client(transport, workspace_id="ws_1")

    assert run(client.retrieve_memory(intent="execute", query="deploy")) == PACKET
    assert path_of(transport.requests[0]) == "/api/retrieve"
    assert json_body(transport.requests[0])["workspace_id"] == "ws_1"


def test_errors_propagate_as_kiomon_error() -> None:
    transport = FakeAsyncTransport(lambda request: json_response({"error": "nope", "code": "insufficient_role"}, 403))
    with pytest.raises(KiomonError) as excinfo:
        run(async_client(transport).search_memory(query="x"))
    assert excinfo.value.code == "insufficient_role"


def test_writes_carry_an_idempotency_key() -> None:
    transport = FakeAsyncTransport(lambda request: json_response({"ok": True}))
    run(async_client(transport, workspace_id="ws_1").draft_memory({"kind": "semantic", "title": "T", "body": "B"}))
    assert header_of(transport.requests[0], "Idempotency-Key")


def test_caller_supplied_options_are_honoured() -> None:
    transport = FakeAsyncTransport(lambda request: json_response({"ok": True}))
    run(
        async_client(transport, workspace_id="ws_1").draft_memory(
            {"kind": "semantic", "title": "T", "body": "B"},
            options=CallOptions(idempotency_key="caller-key"),
        )
    )
    assert header_of(transport.requests[0], "Idempotency-Key") == "caller-key"


def test_select_workspace_chains_and_binds() -> None:
    transport = FakeAsyncTransport(lambda request: json_response({"summary": ""}))
    client = async_client(transport).select_workspace("ws_async")
    assert client.workspace_id == "ws_async"

    run(client.get_workspace_context())
    assert path_of(transport.requests[0]) == "/api/workspaces/ws_async/context"


def test_argument_errors_raise_at_await_time() -> None:
    transport = FakeAsyncTransport()
    with pytest.raises(KiomonError):
        run(async_client(transport).search_memory(query="  "))
    with pytest.raises(KiomonError):
        run(async_client(transport, workspace_id="ws_1").retrieve_memory(intent="plan", query="q", limit=50))
    assert transport.requests == []


def test_async_token_provider_is_awaited_per_request() -> None:
    resolutions: list[int] = []

    async def token() -> str:
        resolutions.append(1)
        return "koa_fresh"

    transport = FakeAsyncTransport(lambda request: json_response({"workspaces": []}))
    client = AsyncKiomon(token=token, base_url="https://api.example.test", transport=transport)

    run(client.list_workspaces())
    run(client.list_workspaces())

    assert header_of(transport.requests[0], "Authorization") == "Bearer koa_fresh"
    assert len(resolutions) == 2


def test_retries_are_recorded_without_real_waiting(async_sleeps: list[float]) -> None:
    attempts = {"n": 0}

    def handler(request: Any) -> Any:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return json_response({"error": "later"}, 503)
        return json_response(SEARCH_OK)

    transport = FakeAsyncTransport(handler)
    assert run(async_client(transport).search_memory(query="x")) == SEARCH_OK
    assert len(transport.requests) == 2
    assert len(async_sleeps) == 1
    assert 0.25 <= async_sleeps[0] <= 0.36


def test_async_context_manager_closes_the_transport() -> None:
    transport = FakeAsyncTransport()

    async def scenario() -> None:
        async with async_client(transport) as client:
            await client.list_workspaces()

    run(scenario())
    assert transport.closed is True


def test_sync_transport_is_rejected_clearly() -> None:
    transport = FakeTransport(lambda request: json_response(SEARCH_OK))
    with pytest.raises(KiomonError) as excinfo:
        run(async_client(transport).list_workspaces())  # type: ignore[arg-type]
    assert excinfo.value.code == "invalid_request"
    assert "async transport" in str(excinfo.value)


def test_repr_is_safe() -> None:
    client = async_client(FakeAsyncTransport())
    assert "sk_kiomon_test" not in repr(client)
    assert "AsyncKiomon" in repr(client)
