"""The listing and graph verbs, and the workspace scoping both depend on.

``list_memories`` is the SDK's half of MCP ``list_memories``: without it a caller
could search but never see what the write gate held back. The graph assertions
are here because the walk is workspace-scoped on the server and its default is
the *credential's* workspace — not the selected one — so a missing
``workspace_id`` reads a different brain and says nothing about it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from conftest import (
    FakeAsyncTransport,
    FakeTransport,
    json_body,
    json_response,
    make_client,
    path_of,
    query_of,
)

from kiomon import AsyncKiomon, KiomonError

MEMORY_PAGE = {
    "memories": [
        {
            "id": "mem_1",
            "title": "Deploys go through the pipeline",
            "excerpt": "Never deploy from a laptop",
            "kind": "semantic",
            "status": "pending_approval",
            "pinned": 0,
            "tags": '["ops"]',
            "source": "agent_reflection",
            "source_url": "manual:eid_1",
            "workspace_id": "ws_1",
            "created_at": "2026-10-17T00:00:00Z",
            "confidence": 0.6,
            "memory_metadata": None,
        }
    ],
    "status": "pending_approval",
    "limit": 50,
    "offset": 0,
    "total": 1,
}


def async_client(transport: Any, **overrides: Any) -> AsyncKiomon:
    options: dict = {"api_key": "sk_kiomon_test", "base_url": "https://api.example.test", "transport": transport}
    options.update(overrides)
    return AsyncKiomon(**options)


# ── list_memories ────────────────────────────────────────────────────────────


class TestListMemories:
    def test_defaults_to_the_review_inbox_and_a_50_row_page(self) -> None:
        transport = FakeTransport(lambda request: json_response(MEMORY_PAGE))
        page = make_client(transport).list_memories()

        assert path_of(transport.requests[0]) == "/api/memories"
        query = query_of(transport.requests[0])
        assert query.get("status") == "pending_approval"
        assert query.get("limit") == "50"
        assert "workspace_id" not in query
        assert page["memories"][0]["id"] == "mem_1"
        assert page["memories"][0]["tags"] == '["ops"]'
        assert page["total"] == 1

    def test_passes_every_filter_through_mapping_query_onto_q(self) -> None:
        transport = FakeTransport(lambda request: json_response({**MEMORY_PAGE, "status": "active"}))
        make_client(transport).list_memories(
            status="active",
            kind="procedural",
            query="deploy",
            limit=10,
            offset=20,
            workspace_id="ws_9",
        )

        query = query_of(transport.requests[0])
        assert query.get("status") == "active"
        assert query.get("kind") == "procedural"
        assert query.get("q") == "deploy"
        assert query.get("limit") == "10"
        assert query.get("offset") == "20"
        assert query.get("workspace_id") == "ws_9"

    def test_falls_back_to_the_selected_workspace(self) -> None:
        transport = FakeTransport(lambda request: json_response(MEMORY_PAGE))
        make_client(transport).select_workspace("ws_selected").list_memories()

        assert query_of(transport.requests[0]).get("workspace_id") == "ws_selected"

    def test_is_a_read_with_no_idempotency_key(self) -> None:
        transport = FakeTransport(lambda request: json_response(MEMORY_PAGE))
        make_client(transport).list_memories()

        assert transport.requests[0].method == "GET"
        assert "Idempotency-Key" not in transport.requests[0].headers

    def test_rejects_a_bad_status_kind_limit_or_offset_before_any_request(self) -> None:
        transport = FakeTransport(lambda request: json_response(MEMORY_PAGE))
        client = make_client(transport)

        for kwargs in (
            {"status": "pending"},
            {"kind": "note"},
            {"limit": 0},
            {"limit": 201},
            {"offset": -1},
        ):
            with pytest.raises(KiomonError):
                client.list_memories(**kwargs)  # type: ignore[arg-type]
        assert transport.requests == []

    def test_async_client_mirrors_the_sync_one(self) -> None:
        transport = FakeAsyncTransport(lambda request: json_response(MEMORY_PAGE))
        page = asyncio.run(async_client(transport).list_memories(status="all", kind="semantic", limit=5))

        query = query_of(transport.requests[0])
        assert query.get("status") == "all"
        assert query.get("kind") == "semantic"
        assert query.get("limit") == "5"
        assert page["memories"][0]["id"] == "mem_1"


# ── explore_memory_graph ─────────────────────────────────────────────────────


class TestExploreMemoryGraph:
    def test_scopes_the_walk_to_the_selected_workspace(self) -> None:
        transport = FakeTransport(lambda request: json_response({"nodes": [], "edges": []}))
        make_client(transport).select_workspace("ws_selected").explore_memory_graph(id="mem_1")

        query = query_of(transport.requests[0])
        assert query.get("center_id") == "mem_1"
        assert query.get("depth") == "1"
        assert query.get("workspace_id") == "ws_selected"

    def test_lets_an_explicit_workspace_override_the_selection(self) -> None:
        transport = FakeTransport(lambda request: json_response({"nodes": [], "edges": []}))
        make_client(transport).select_workspace("ws_selected").explore_memory_graph(
            id="mem_1", relation="contradicts", depth=2, workspace_id="ws_other"
        )

        query = query_of(transport.requests[0])
        assert query.get("workspace_id") == "ws_other"
        assert query.get("relation") == "contradicts"
        assert query.get("depth") == "2"

    def test_async_client_scopes_it_too(self) -> None:
        transport = FakeAsyncTransport(lambda request: json_response({"nodes": [], "edges": []}))
        asyncio.run(async_client(transport).select_workspace("ws_selected").explore_memory_graph(id="mem_1"))

        assert query_of(transport.requests[0]).get("workspace_id") == "ws_selected"


# ── record_outcome ───────────────────────────────────────────────────────────


class TestRecordOutcome:
    def test_sends_the_assessment_vocabulary_the_server_accepts(self) -> None:
        transport = FakeTransport(lambda request: json_response({"updated": ["mem_1"], "skipped": []}))
        make_client(transport).record_outcome(
            outcome="success",
            items=[
                {"memory_id": "mem_1", "assessment": "helpful"},
                {"memory_id": "mem_2", "assessment": "outdated"},
            ],
        )

        assert json_body(transport.requests[0])["items"] == [
            {"memory_id": "mem_1", "assessment": "helpful"},
            {"memory_id": "mem_2", "assessment": "outdated"},
        ]
