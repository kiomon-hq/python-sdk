"""The SDK-only verbs that make the memory verbs usable from a cold start.

``create_workspace``, ``ingest_document``, ``list_documents`` and
``delete_document`` have no MCP equivalent — they exist because without them a
caller cannot make the workspace ``select_workspace`` needs, nor put anything in
it for ``search_memory`` to find.

The ``User-Agent`` assertions live here too: the header is what keeps
Cloudflare's edge from refusing every request with a 403, so it is part of the
same "can this SDK be used at all" contract.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from conftest import (
    FakeAsyncTransport,
    FakeTransport,
    header_of,
    json_body,
    json_response,
    make_client,
    path_of,
    query_of,
)

from kiomon import AsyncKiomon, KiomonError
from kiomon._specs import USER_AGENT

WORKSPACE = {"id": "ws_new", "user_id": "u_1", "name": "W", "slug": "w-abc123", "org_id": "org_1", "created_at": "t"}
DOCUMENT_LIST = {"documents": [], "limit": 20, "offset": 0, "total": 0, "counts": {}}
INGESTED = {"id": "doc_1", "tags": ["a"]}
DELETED = {"deleted": True}


def async_client(transport: Any, **overrides: Any) -> AsyncKiomon:
    options: dict = {"api_key": "sk_kiomon_test", "base_url": "https://api.example.test", "transport": transport}
    options.update(overrides)
    return AsyncKiomon(**options)


# ── User-Agent ───────────────────────────────────────────────────────────────


class TestUserAgent:
    """Without this header Cloudflare answers 403 to every request.

    The default transport sets none of its own, so urllib's `Python-urllib/3.x`
    goes out and the edge bans it (Error 1010, browser_signature_banned). A test
    that asserts the header is present is the only thing standing between that
    and a published SDK that cannot talk to its own API.
    """

    def test_every_request_identifies_the_sdk(self) -> None:
        transport = FakeTransport(lambda request: json_response({"workspaces": []}))
        make_client(transport).list_workspaces()
        assert header_of(transport.requests[0], "User-Agent") == USER_AGENT

    def test_carries_the_package_version(self) -> None:
        import kiomon

        assert f"kiomon-python-sdk/{kiomon.__version__}" == USER_AGENT

    def test_a_caller_can_override_it(self) -> None:
        transport = FakeTransport(lambda request: json_response({"workspaces": []}))
        make_client(transport, headers={"User-Agent": "my-app/2.0"}).list_workspaces()
        assert header_of(transport.requests[0], "User-Agent") == "my-app/2.0"

    def test_the_async_client_sends_it_too(self) -> None:
        transport = FakeAsyncTransport(lambda request: json_response({"workspaces": []}))
        asyncio.run(async_client(transport).list_workspaces())
        assert header_of(transport.requests[0], "User-Agent") == USER_AGENT


# ── create_workspace ─────────────────────────────────────────────────────────


class TestCreateWorkspace:
    def test_posts_to_the_workspace_route_with_an_idempotency_key(self) -> None:
        transport = FakeTransport(lambda request: json_response(WORKSPACE, 201))
        make_client(transport).create_workspace(name="Research")

        request = transport.requests[0]
        assert request.method == "POST"
        assert path_of(request) == "/api/workspaces"
        assert json_body(request) == {"name": "Research"}
        assert header_of(request, "Idempotency-Key") is not None

    def test_places_the_workspace_in_an_org_when_asked(self) -> None:
        transport = FakeTransport(lambda request: json_response(WORKSPACE, 201))
        make_client(transport).create_workspace(name="Research", org_id="org_1")
        assert json_body(transport.requests[0]) == {"name": "Research", "org_id": "org_1"}

    def test_omits_org_id_rather_than_sending_null(self) -> None:
        """`org_id: null` would read as an explicit org on the wire; absent is not the same."""
        transport = FakeTransport(lambda request: json_response(WORKSPACE, 201))
        make_client(transport).create_workspace(name="Research", org_id=None)
        assert "org_id" not in json_body(transport.requests[0])

    def test_rejects_an_empty_name_without_a_request(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError) as excinfo:
            make_client(transport).create_workspace(name="   ")
        assert excinfo.value.code == "invalid_request"
        assert transport.requests == []

    def test_rejects_an_empty_org_id(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError, match="org_id"):
            make_client(transport).create_workspace(name="W", org_id="")
        assert transport.requests == []

    def test_a_free_plan_ceiling_surfaces_as_quota_exceeded(self) -> None:
        transport = FakeTransport(
            lambda request: json_response({"error": "Workspace limit reached.", "code": "quota_exceeded"}, 402)
        )
        with pytest.raises(KiomonError) as excinfo:
            make_client(transport).create_workspace(name="W")
        assert excinfo.value.code == "quota_exceeded"

    def test_the_async_client_creates_one_too(self) -> None:
        transport = FakeAsyncTransport(lambda request: json_response(WORKSPACE, 201))
        result = asyncio.run(async_client(transport).create_workspace(name="Research", org_id="org_1"))
        assert result["id"] == "ws_new"
        assert json_body(transport.requests[0]) == {"name": "Research", "org_id": "org_1"}


# ── ingest_document ──────────────────────────────────────────────────────────


class TestIngestDocument:
    def test_posts_the_body_to_the_document_route(self) -> None:
        transport = FakeTransport(lambda request: json_response(INGESTED, 201))
        result = make_client(transport).ingest_document(body="# Title\n\nText.")

        request = transport.requests[0]
        assert request.method == "POST"
        assert path_of(request) == "/api/documents"
        assert json_body(request) == {"body": "# Title\n\nText."}
        assert result["id"] == "doc_1"

    def test_carries_the_optional_metadata(self) -> None:
        transport = FakeTransport(lambda request: json_response(INGESTED, 201))
        make_client(transport).ingest_document(
            body="b",
            title="T",
            source_url="https://example.test/a",
            tags=["runbook", "deploy"],
            kind="procedural",
        )
        assert json_body(transport.requests[0]) == {
            "body": "b",
            "title": "T",
            "source_url": "https://example.test/a",
            "tags": ["runbook", "deploy"],
            "kind": "procedural",
        }

    def test_omits_unset_optional_fields(self) -> None:
        """An explicit `null` would clear a field the caller never mentioned."""
        transport = FakeTransport(lambda request: json_response(INGESTED, 201))
        make_client(transport).ingest_document(body="b", title=None, author=None, tags=None)
        assert json_body(transport.requests[0]) == {"body": "b"}

    def test_the_bound_workspace_is_attached(self) -> None:
        transport = FakeTransport(lambda request: json_response(INGESTED, 201))
        make_client(transport, workspace_id="ws_bound").ingest_document(body="b")
        assert json_body(transport.requests[0])["workspace_id"] == "ws_bound"

    def test_an_explicit_workspace_wins_over_the_bound_one(self) -> None:
        transport = FakeTransport(lambda request: json_response(INGESTED, 201))
        make_client(transport, workspace_id="ws_bound").ingest_document(body="b", workspace_id="ws_explicit")
        assert json_body(transport.requests[0])["workspace_id"] == "ws_explicit"

    def test_source_url_is_what_makes_a_second_save_a_no_op(self) -> None:
        """Dedup keys on source_url alone; without one the API mints a unique synthetic URL."""
        transport = FakeTransport(lambda request: json_response(INGESTED, 201))
        make_client(transport).ingest_document(body="b", source_url="https://example.test/a")
        assert json_body(transport.requests[0])["source_url"] == "https://example.test/a"

    def test_a_duplicate_is_reported_rather_than_hidden(self) -> None:
        """The API answers 200 + the existing id; the SDK must not present it as new."""
        transport = FakeTransport(
            lambda request: json_response({"duplicate": True, "id": "doc_old", "created_at": "t"}, 200)
        )
        result = make_client(transport).ingest_document(body="b")
        assert result["duplicate"] is True
        assert result["id"] == "doc_old"

    def test_rejects_an_empty_body_without_a_request(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError) as excinfo:
            make_client(transport).ingest_document(body="")
        assert excinfo.value.code == "invalid_request"
        assert transport.requests == []

    def test_rejects_an_unknown_kind(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError, match="kind"):
            make_client(transport).ingest_document(body="b", kind="not-a-kind")  # type: ignore[arg-type]
        assert transport.requests == []

    def test_rejects_non_string_tags(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError, match="tags"):
            make_client(transport).ingest_document(body="b", tags=["ok", 3])  # type: ignore[list-item]
        assert transport.requests == []

    def test_the_async_client_ingests_too(self) -> None:
        transport = FakeAsyncTransport(lambda request: json_response(INGESTED, 201))
        result = asyncio.run(async_client(transport, workspace_id="ws_1").ingest_document(body="b", title="T"))
        assert result["id"] == "doc_1"
        assert json_body(transport.requests[0]) == {"body": "b", "title": "T", "workspace_id": "ws_1"}


# ── list_documents ───────────────────────────────────────────────────────────


class TestListDocuments:
    def test_reads_the_document_route_with_no_filters(self) -> None:
        transport = FakeTransport(lambda request: json_response(DOCUMENT_LIST))
        make_client(transport).list_documents()

        request = transport.requests[0]
        assert request.method == "GET"
        assert path_of(request) == "/api/documents"
        assert query_of(request) == {}

    def test_maps_every_filter_onto_the_query_string(self) -> None:
        transport = FakeTransport(lambda request: json_response(DOCUMENT_LIST))
        make_client(transport).list_documents(
            query="deploy",
            kind="procedural",
            tag="runbook",
            status="active",
            source="user_saved",
            sort="oldest",
            since="2026-01-01",
            index_status="all",
            limit=5,
            offset=10,
        )
        assert query_of(transport.requests[0]) == {
            "q": "deploy",
            "kind": "procedural",
            "tag": "runbook",
            "status": "active",
            "source": "user_saved",
            "sort": "oldest",
            "since": "2026-01-01",
            "index_status": "all",
            "limit": "5",
            "offset": "10",
        }

    def test_the_bound_workspace_is_attached(self) -> None:
        transport = FakeTransport(lambda request: json_response(DOCUMENT_LIST))
        make_client(transport, workspace_id="ws_bound").list_documents()
        assert query_of(transport.requests[0])["workspace_id"] == "ws_bound"

    def test_returns_the_page_and_the_counts(self) -> None:
        payload = {
            "documents": [{"id": "doc_1", "title": "T", "kind": "semantic", "rank": 0.4}],
            "limit": 20,
            "offset": 0,
            "total": 12,
            "counts": {"total": 12, "semantic": 9, "episodic": 1, "procedural": 2, "reference": 0, "pinned": 0},
        }
        transport = FakeTransport(lambda request: json_response(payload))
        result = make_client(transport).list_documents()
        assert result["total"] == 12
        assert result["counts"]["semantic"] == 9

    @pytest.mark.parametrize("limit", [0, 201, -1])
    def test_enforces_the_page_ceiling_locally(self, limit: int) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError) as excinfo:
            make_client(transport).list_documents(limit=limit)
        assert excinfo.value.code == "invalid_request"
        assert transport.requests == []

    def test_rejects_a_negative_offset(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError, match="offset"):
            make_client(transport).list_documents(offset=-1)
        assert transport.requests == []

    @pytest.mark.parametrize("status", ["deleted", "ACTIVE", ""])
    def test_rejects_an_unknown_status(self, status: str) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError, match="status"):
            make_client(transport).list_documents(status=status)  # type: ignore[arg-type]
        assert transport.requests == []

    def test_rejects_an_unknown_sort(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError, match="sort"):
            make_client(transport).list_documents(sort="newest")  # type: ignore[arg-type]
        assert transport.requests == []

    def test_rejects_an_unknown_kind(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError, match="kind"):
            make_client(transport).list_documents(kind="note")  # type: ignore[arg-type]
        assert transport.requests == []

    def test_index_status_is_omitted_by_default(self) -> None:
        """Absent means "findable only", which is the server's default, not "all"."""
        transport = FakeTransport(lambda request: json_response(DOCUMENT_LIST))
        make_client(transport).list_documents()
        assert "index_status" not in query_of(transport.requests[0])

    def test_index_status_all_reaches_the_wire(self) -> None:
        transport = FakeTransport(lambda request: json_response(DOCUMENT_LIST))
        make_client(transport).list_documents(index_status="all")
        assert query_of(transport.requests[0])["index_status"] == "all"

    @pytest.mark.parametrize("index_status", ["pending", "READY", "indexing"])
    def test_rejects_an_unknown_index_status(self, index_status: str) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError, match="index_status"):
            make_client(transport).list_documents(index_status=index_status)  # type: ignore[arg-type]
        assert transport.requests == []

    def test_the_async_client_lists_too(self) -> None:
        transport = FakeAsyncTransport(lambda request: json_response(DOCUMENT_LIST))
        asyncio.run(async_client(transport).list_documents(query="x", limit=5))
        assert query_of(transport.requests[0]) == {"q": "x", "limit": "5"}


# ── delete_document ──────────────────────────────────────────────────────────


class TestDeleteDocument:
    def test_deletes_the_document_route(self) -> None:
        transport = FakeTransport(lambda request: json_response(DELETED))
        result = make_client(transport).delete_document("doc_1")

        request = transport.requests[0]
        assert request.method == "DELETE"
        assert path_of(request) == "/api/documents/doc_1"
        assert result["deleted"] is True

    def test_escapes_an_id_that_needs_it(self) -> None:
        transport = FakeTransport(lambda request: json_response(DELETED))
        make_client(transport).delete_document("doc/../1")
        assert path_of(transport.requests[0]) == "/api/documents/doc%2F..%2F1"

    def test_rejects_an_empty_id_without_a_request(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError) as excinfo:
            make_client(transport).delete_document("")
        assert excinfo.value.code == "invalid_request"
        assert transport.requests == []

    def test_the_async_client_deletes_too(self) -> None:
        transport = FakeAsyncTransport(lambda request: json_response(DELETED))
        result = asyncio.run(async_client(transport).delete_document("doc_1"))
        assert result["deleted"] is True
        assert path_of(transport.requests[0]) == "/api/documents/doc_1"


# ── The dead end these verbs remove ──────────────────────────────────────────


def test_no_workspace_points_at_create_workspace_not_the_dashboard() -> None:
    """The old message sent SDK-only users out of the SDK, which is the gap these verbs close."""
    transport = FakeTransport(lambda request: json_response({"workspaces": []}))
    with pytest.raises(KiomonError) as excinfo:
        make_client(transport).get_workspace_context()
    assert "create_workspace()" in str(excinfo.value)
    assert "dashboard" not in str(excinfo.value)
