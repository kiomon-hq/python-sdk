"""Client behaviour: verbs, workspace resolution, errors, retries, validation."""

from __future__ import annotations

from typing import Any

import pytest
from conftest import (
    FakeTransport,
    header_of,
    json_body,
    json_response,
    make_client,
    path_of,
    query_of,
)

from kiomon import CallOptions, Kiomon, KiomonError
from kiomon._transport import Response

LEARNINGS = [{"kind": "semantic", "title": "T", "body": "B"}]
SEARCH_OK = {"results": []}
BATCH_OK = {"drafted": 1, "promoted": [], "pending": [], "memories": []}


# ── Construction ─────────────────────────────────────────────────────────────


class TestConstruction:
    def test_requires_credentials(self) -> None:
        with pytest.raises(KiomonError) as excinfo:
            Kiomon(transport=FakeTransport())
        assert excinfo.value.code == "invalid_request"

    def test_rejects_both_credential_types(self) -> None:
        with pytest.raises(KiomonError, match="not both"):
            Kiomon(api_key="a", token="b", transport=FakeTransport())

    def test_rejects_bad_timeout_and_retries(self) -> None:
        with pytest.raises(KiomonError):
            Kiomon(api_key="a", timeout=0, transport=FakeTransport())
        with pytest.raises(KiomonError):
            Kiomon(api_key="a", max_retries=-1, transport=FakeTransport())

    def test_sends_api_key_as_x_api_key(self) -> None:
        transport = FakeTransport(lambda request: json_response(SEARCH_OK))
        make_client(transport).search_memory(query="x")
        assert header_of(transport.requests[0], "X-API-Key") == "sk_kiomon_test"
        assert header_of(transport.requests[0], "Authorization") is None

    def test_sends_delegated_token_as_bearer_and_resolves_it_per_request(self) -> None:
        resolutions: list[int] = []

        def token() -> str:
            resolutions.append(1)
            return "koa_fresh"

        transport = FakeTransport(lambda request: json_response({"workspaces": []}))
        client = Kiomon(token=token, base_url="https://api.example.test", transport=transport)

        client.list_workspaces()
        client.list_workspaces()

        assert header_of(transport.requests[0], "Authorization") == "Bearer koa_fresh"
        assert len(resolutions) == 2

    def test_token_provider_returning_nothing_is_unauthorized(self) -> None:
        transport = FakeTransport(lambda request: json_response({"workspaces": []}))
        client = Kiomon(token=lambda: None, base_url="https://api.example.test", transport=transport)
        with pytest.raises(KiomonError) as excinfo:
            client.list_workspaces()
        assert excinfo.value.code == "unauthorized"
        assert transport.requests == []

    def test_identifies_itself_in_telemetry(self) -> None:
        transport = FakeTransport(lambda request: json_response(SEARCH_OK))
        make_client(transport).search_memory(query="x")
        assert header_of(transport.requests[0], "X-Agent-Id") == "sdk-python"

        overridden = FakeTransport(lambda request: json_response(SEARCH_OK))
        make_client(overridden, agent_id="my-agent").search_memory(query="x")
        assert header_of(overridden.requests[0], "X-Agent-Id") == "my-agent"

    def test_extra_headers_are_sent(self) -> None:
        transport = FakeTransport(lambda request: json_response(SEARCH_OK))
        make_client(transport, headers={"X-Custom": "1"}).search_memory(query="x")
        assert header_of(transport.requests[0], "X-Custom") == "1"

    def test_base_url_trailing_slash_is_stripped(self) -> None:
        transport = FakeTransport(lambda request: json_response(SEARCH_OK))
        make_client(transport, base_url="https://api.example.test/").search_memory(query="x")
        assert transport.requests[0].url.startswith("https://api.example.test/api/search")

    def test_repr_does_not_leak_the_key(self) -> None:
        client = make_client(FakeTransport())
        assert "sk_kiomon_test" not in repr(client)
        assert "api_key" in repr(client)


# ── Verbs map to the documented REST surface ─────────────────────────────────


class TestVerbs:
    def test_search_builds_a_query_string_and_omits_an_unset_workspace(self) -> None:
        transport = FakeTransport(lambda request: json_response(SEARCH_OK))
        make_client(transport).search_memory(query="vector db", kind="reference", limit=3)

        request = transport.requests[0]
        assert path_of(request) == "/api/search"
        assert query_of(request) == {"q": "vector db", "kind": "reference", "limit": "3"}
        assert "workspace_id" not in query_of(request)

    def test_search_defaults_limit_to_10(self) -> None:
        transport = FakeTransport(lambda request: json_response(SEARCH_OK))
        make_client(transport).search_memory(query="x")
        assert query_of(transport.requests[0])["limit"] == "10"

    def test_search_accepts_several_kinds(self) -> None:
        transport = FakeTransport(lambda request: json_response(SEARCH_OK))
        make_client(transport).search_memory(query="x", kind=["semantic", "procedural"])
        assert query_of(transport.requests[0])["kind"] == "semantic,procedural"

    def test_search_includes_the_configured_workspace(self) -> None:
        transport = FakeTransport(lambda request: json_response(SEARCH_OK))
        make_client(transport, workspace_id="ws_1").search_memory(query="x")
        assert query_of(transport.requests[0])["workspace_id"] == "ws_1"

    def test_get_memory_targets_the_document_route(self) -> None:
        transport = FakeTransport(lambda request: json_response({"id": "doc_1"}))
        assert make_client(transport).get_memory("doc_1") == {"id": "doc_1"}
        assert path_of(transport.requests[0]) == "/api/documents/doc_1"

    def test_get_memory_escapes_the_id(self) -> None:
        transport = FakeTransport(lambda request: json_response({"id": "a/b"}))
        make_client(transport).get_memory("a/b")
        assert path_of(transport.requests[0]) == "/api/documents/a%2Fb"

    def test_get_workspace_context_targets_the_context_route(self) -> None:
        transport = FakeTransport(lambda request: json_response({"summary": ""}))
        make_client(transport, workspace_id="ws_1").get_workspace_context(topic="deploy")

        request = transport.requests[0]
        assert path_of(request) == "/api/workspaces/ws_1/context"
        assert query_of(request) == {"topic": "deploy"}

    def test_get_latest_briefing_targets_the_briefing_route(self) -> None:
        transport = FakeTransport(lambda request: json_response({"id": "b1"}))
        make_client(transport, workspace_id="ws_1").get_latest_briefing()
        assert path_of(transport.requests[0]) == "/api/workspaces/ws_1/briefings/latest"

    def test_explore_graph_defaults_to_one_hop(self) -> None:
        transport = FakeTransport(lambda request: json_response({"nodes": [], "edges": []}))
        make_client(transport).explore_memory_graph(id="m1")
        assert path_of(transport.requests[0]) == "/api/graph"
        assert query_of(transport.requests[0]) == {"center_id": "m1", "depth": "1"}

    def test_explore_graph_passes_relation_and_depth(self) -> None:
        transport = FakeTransport(lambda request: json_response({"nodes": [], "edges": []}))
        make_client(transport).explore_memory_graph(id="m1", relation="supports", depth=2)
        assert query_of(transport.requests[0]) == {"center_id": "m1", "depth": "2", "relation": "supports"}

    def test_list_workspaces_targets_the_workspace_route(self) -> None:
        transport = FakeTransport(lambda request: json_response({"workspaces": []}))
        assert make_client(transport).list_workspaces() == {"workspaces": []}
        assert path_of(transport.requests[0]) == "/api/workspaces"

    def test_reflect_session_posts_learnings_to_the_batch_draft_route(self) -> None:
        transport = FakeTransport(lambda request: json_response(BATCH_OK))
        make_client(transport, workspace_id="ws_1").reflect_session(learnings=LEARNINGS)

        request = transport.requests[0]
        assert request.method == "POST"
        assert path_of(request) == "/api/memories/batch-draft"
        assert json_body(request) == {"workspace_id": "ws_1", "learnings": LEARNINGS}

    def test_draft_memory_is_sugar_over_reflect_session(self) -> None:
        transport = FakeTransport(lambda request: json_response(BATCH_OK))
        make_client(transport, workspace_id="ws_1").draft_memory({"kind": "semantic", "title": "T", "body": "B"})

        request = transport.requests[0]
        assert path_of(request) == "/api/memories/batch-draft"
        assert json_body(request) == {"workspace_id": "ws_1", "learnings": LEARNINGS}

    def test_draft_memory_drops_unset_optional_fields(self) -> None:
        transport = FakeTransport(lambda request: json_response(BATCH_OK))
        make_client(transport, workspace_id="ws_1").draft_memory(
            {
                "kind": "semantic",
                "title": "T",
                "body": "B",
                "excerpt": None,
                "tags": ["a"],
                "confidence": 0.9,
            }
        )
        learnings = json_body(transport.requests[0])["learnings"]
        assert learnings == [
            {"kind": "semantic", "title": "T", "body": "B", "tags": ["a"], "confidence": 0.9}
        ]

    def test_retrieve_memory_posts_the_intent_packet_request(self) -> None:
        transport = FakeTransport(lambda request: json_response({"summary": "", "facts": [], "episodes": []}))
        make_client(transport, workspace_id="ws_1").retrieve_memory(intent="execute", query="deploy")

        request = transport.requests[0]
        assert request.method == "POST"
        assert path_of(request) == "/api/retrieve"
        assert json_body(request) == {"intent": "execute", "query": "deploy", "workspace_id": "ws_1"}

    def test_retrieve_memory_forwards_every_optional_field(self) -> None:
        transport = FakeTransport(lambda request: json_response({"summary": ""}))
        make_client(transport).retrieve_memory(
            intent="debug",
            query="q",
            entities=["a"],
            constraints=["c"],
            kinds=["semantic"],
            risk_level="high",
            limit=4,
        )
        assert json_body(transport.requests[0]) == {
            "intent": "debug",
            "query": "q",
            "entities": ["a"],
            "constraints": ["c"],
            "kinds": ["semantic"],
            "risk_level": "high",
            "limit": 4,
        }

    def test_record_outcome_posts_items(self) -> None:
        transport = FakeTransport(lambda request: json_response({"updated": []}))
        items = [{"memory_id": "m1", "assessment": "helpful"}]
        make_client(transport, workspace_id="ws_1").record_outcome(outcome="success", items=items)

        request = transport.requests[0]
        assert request.method == "POST"
        assert path_of(request) == "/api/memories/record-outcome"
        assert json_body(request) == {"outcome": "success", "items": items, "workspace_id": "ws_1"}

    @pytest.mark.parametrize(
        ("action", "value", "method", "path", "body"),
        [
            ("pin", None, "PATCH", "/api/documents/m1", {"pinned": 1}),
            ("unpin", None, "PATCH", "/api/documents/m1", {"pinned": 0}),
            ("archive", None, "POST", "/api/documents/m1/archive", None),
            ("restore", None, "POST", "/api/documents/m1/restore", None),
            ("forget", None, "POST", "/api/documents/m1/forget", None),
            ("approve", None, "POST", "/api/documents/m1/approve", None),
            ("reject", None, "POST", "/api/documents/m1/reject", None),
            ("rate", 1, "POST", "/api/documents/m1/rate", {"rating": 1}),
            ("rate", -1, "POST", "/api/documents/m1/rate", {"rating": -1}),
            ("revert", None, "POST", "/api/memories/m1/revert", None),
        ],
    )
    def test_manage_memory_dispatches_each_action(
        self, action: str, value: Any, method: str, path: str, body: Any
    ) -> None:
        transport = FakeTransport(lambda request: json_response({"ok": True}))
        make_client(transport).manage_memory(id="m1", action=action, value=value)  # type: ignore[arg-type]

        request = transport.requests[0]
        assert request.method == method
        assert path_of(request) == path
        assert json_body(request) == body

    def test_request_reaches_any_route(self) -> None:
        transport = FakeTransport(lambda request: json_response({"anything": True}))
        assert make_client(transport).request("/api/user/stats") == {"anything": True}
        assert path_of(transport.requests[0]) == "/api/user/stats"


# ── Workspace resolution ─────────────────────────────────────────────────────


class TestWorkspaces:
    def test_resolves_a_single_workspace_silently_then_caches_it(self) -> None:
        def handler(request: Any) -> Response:
            if path_of(request) == "/api/workspaces":
                return json_response({"workspaces": [{"id": "ws_only", "slug": "solo"}]})
            return json_response({"summary": "context"})

        transport = FakeTransport(handler)
        client = make_client(transport)

        client.get_workspace_context()
        client.get_workspace_context()

        assert client.workspace_id == "ws_only"
        lookups = [r for r in transport.requests if path_of(r) == "/api/workspaces"]
        assert len(lookups) == 1

    def test_refuses_to_guess_when_several_workspaces_exist(self) -> None:
        transport = FakeTransport(
            lambda request: json_response(
                {"workspaces": [{"id": "ws_a", "slug": "alpha"}, {"id": "ws_b", "slug": "beta"}]}
            )
        )
        with pytest.raises(KiomonError) as excinfo:
            make_client(transport).get_latest_briefing()

        assert excinfo.value.code == "invalid_request"
        assert "ws_a" in str(excinfo.value)
        assert "beta" in str(excinfo.value)

    def test_no_workspaces_is_not_found(self) -> None:
        transport = FakeTransport(lambda request: json_response({"workspaces": []}))
        with pytest.raises(KiomonError) as excinfo:
            make_client(transport).get_workspace_context()
        assert excinfo.value.code == "not_found"

    def test_select_workspace_binds_the_client_and_chains(self) -> None:
        transport = FakeTransport(lambda request: json_response({"summary": ""}))
        client = make_client(transport).select_workspace("ws_chosen")
        client.get_workspace_context()
        assert path_of(transport.requests[0]) == "/api/workspaces/ws_chosen/context"


# ── Errors ───────────────────────────────────────────────────────────────────


class TestErrors:
    def test_surfaces_the_api_message_and_marks_quota_rejections_upgradeable(self) -> None:
        transport = FakeTransport(
            lambda request: json_response({"error": "Free plan limit reached", "needsUpgrade": True}, 402)
        )
        with pytest.raises(KiomonError) as excinfo:
            make_client(transport).search_memory(query="x")

        error = excinfo.value
        assert error.code == "quota_exceeded"
        assert error.needs_upgrade is True
        assert str(error) == "Free plan limit reached"
        assert error.retryable is False

    def test_passes_a_server_supplied_code_through_even_when_unknown(self) -> None:
        transport = FakeTransport(lambda request: json_response({"error": "nope", "code": "brand_new"}, 400))
        with pytest.raises(KiomonError) as excinfo:
            make_client(transport).search_memory(query="x")
        assert excinfo.value.code == "brand_new"

    def test_keeps_a_404_meaningful_since_denial_also_returns_404(self) -> None:
        transport = FakeTransport(lambda request: json_response({"error": "Not found"}, 404))
        with pytest.raises(KiomonError) as excinfo:
            make_client(transport).get_memory("m_missing")
        assert excinfo.value.code == "not_found"
        assert excinfo.value.status == 404

    def test_keeps_the_request_id_from_the_response(self) -> None:
        transport = FakeTransport(
            lambda request: json_response({"error": "nope"}, 500, headers={"x-request-id": "req_123"})
        )
        with pytest.raises(KiomonError) as excinfo:
            make_client(transport).search_memory(query="x")
        assert excinfo.value.request_id == "req_123"

    def test_a_non_json_success_body_is_a_retryable_service_error(self, sleeps: list[float]) -> None:
        transport = FakeTransport(lambda request: Response(200, {"content-type": "text/html"}, b"<html>nope</html>"))
        with pytest.raises(KiomonError) as excinfo:
            make_client(transport).request("/api/anything")

        assert excinfo.value.code == "service_unavailable"
        assert excinfo.value.retryable is True
        assert len(transport.requests) == 3  # unparseable response is retried like a 5xx

    def test_empty_success_response_is_none(self) -> None:
        transport = FakeTransport(lambda request: Response(204, {}, b""))
        assert make_client(transport).request("/api/anything") is None

    def test_does_not_retry_an_auth_failure(self, sleeps: list[float]) -> None:
        transport = FakeTransport(lambda request: json_response({"error": "bad key"}, 401))
        with pytest.raises(KiomonError):
            make_client(transport).search_memory(query="x")
        assert len(transport.requests) == 1
        assert sleeps == []


# ── Retries and idempotency ──────────────────────────────────────────────────


class TestRetries:
    def test_retries_a_rate_limited_read_and_succeeds(self, sleeps: list[float]) -> None:
        attempts = {"n": 0}

        def handler(request: Any) -> Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return json_response({"error": "slow down"}, 429)
            return json_response(SEARCH_OK)

        transport = FakeTransport(handler)
        assert make_client(transport).search_memory(query="x") == SEARCH_OK
        assert len(transport.requests) == 2
        assert len(sleeps) == 1

    def test_retries_a_network_failure(self, sleeps: list[float]) -> None:
        attempts = {"n": 0}

        def handler(request: Any) -> Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise KiomonError("connection reset", code="network_error")
            return json_response(SEARCH_OK)

        transport = FakeTransport(handler)
        assert make_client(transport).search_memory(query="x") == SEARCH_OK
        assert len(transport.requests) == 2

    def test_a_client_error_is_not_retried(self, sleeps: list[float]) -> None:
        transport = FakeTransport(lambda request: json_response({"error": "bad"}, 400))
        with pytest.raises(KiomonError):
            make_client(transport).search_memory(query="x")
        assert len(transport.requests) == 1

    def test_max_retries_can_be_overridden_per_call(self) -> None:
        transport = FakeTransport(lambda request: json_response({"error": "later"}, 503))
        client = make_client(transport)
        with pytest.raises(KiomonError):
            client.search_memory(query="x", options=CallOptions(max_retries=0))
        assert len(transport.requests) == 1

    def test_retry_after_is_honoured(self, sleeps: list[float]) -> None:
        attempts = {"n": 0}

        def handler(request: Any) -> Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return json_response({"error": "slow", "retry_after": 2.0}, 429)
            return json_response(SEARCH_OK)

        make_client(FakeTransport(handler)).search_memory(query="x")
        assert sleeps == [2.0]

    def test_backoff_is_exponential_with_jitter(self, sleeps: list[float]) -> None:
        attempts = {"n": 0}

        def handler(request: Any) -> Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return json_response({"error": "later"}, 503)
            return json_response(SEARCH_OK)

        make_client(FakeTransport(handler)).search_memory(query="x")
        assert len(sleeps) == 1
        assert 0.25 <= sleeps[0] <= 0.36

    def test_reads_carry_no_idempotency_key(self) -> None:
        transport = FakeTransport(lambda request: json_response(SEARCH_OK))
        client = make_client(transport, workspace_id="ws_1")
        client.search_memory(query="x")
        assert header_of(transport.requests[0], "Idempotency-Key") is None

    def test_a_reading_post_carries_no_idempotency_key(self) -> None:
        transport = FakeTransport(lambda request: json_response({"summary": ""}))
        make_client(transport, workspace_id="ws_1").retrieve_memory(intent="plan", query="q")

        assert transport.requests[0].method == "POST"
        assert header_of(transport.requests[0], "Idempotency-Key") is None

    def test_write_retries_reuse_one_idempotency_key(self, sleeps: list[float]) -> None:
        attempts = {"n": 0}

        def handler(request: Any) -> Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return json_response({"error": "later"}, 503)
            return json_response(BATCH_OK)

        transport = FakeTransport(handler)
        make_client(transport, workspace_id="ws_1").reflect_session(learnings=LEARNINGS)

        assert len(transport.requests) == 2
        first = header_of(transport.requests[0], "Idempotency-Key")
        assert first
        assert header_of(transport.requests[1], "Idempotency-Key") == first

    def test_a_caller_supplied_key_is_used_verbatim(self) -> None:
        transport = FakeTransport(lambda request: json_response({"updated": []}))
        make_client(transport).record_outcome(
            outcome="success",
            items=[{"memory_id": "m1", "assessment": "helpful"}],
            options=CallOptions(idempotency_key="caller-key-1"),
        )
        assert header_of(transport.requests[0], "Idempotency-Key") == "caller-key-1"

    def test_request_is_a_write_only_when_declared(self) -> None:
        undeclared = FakeTransport(lambda request: json_response({"ok": True}))
        make_client(undeclared).request("/api/documents", method="POST", body={"a": 1})
        assert header_of(undeclared.requests[0], "Idempotency-Key") is None

        declared = FakeTransport(lambda request: json_response({"ok": True}))
        make_client(declared).request("/api/documents", method="POST", body={"a": 1}, write=True)
        assert header_of(declared.requests[0], "Idempotency-Key")

    def test_request_id_is_forwarded(self) -> None:
        transport = FakeTransport(lambda request: json_response(SEARCH_OK))
        make_client(transport).search_memory(query="x", options=CallOptions(request_id="req_abc"))
        assert header_of(transport.requests[0], "X-Request-Id") == "req_abc"


# ── Input validation happens locally ─────────────────────────────────────────


class TestValidation:
    def test_rejects_an_empty_search_query_without_a_request(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError, match="non-empty"):
            make_client(transport).search_memory(query="  ")
        assert transport.requests == []

    def test_rejects_an_empty_memory_id(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError):
            make_client(transport).get_memory("")
        with pytest.raises(KiomonError):
            make_client(transport).explore_memory_graph(id=" ")
        assert transport.requests == []

    def test_rejects_rate_without_a_value_of_1_or_minus_1(self) -> None:
        transport = FakeTransport()
        client = make_client(transport)
        with pytest.raises(KiomonError, match="1 or -1"):
            client.manage_memory(id="m1", action="rate")
        with pytest.raises(KiomonError, match="1 or -1"):
            client.manage_memory(id="m1", action="rate", value=0)
        with pytest.raises(KiomonError, match="1 or -1"):
            client.manage_memory(id="m1", action="rate", value=True)  # type: ignore[arg-type]
        assert transport.requests == []

    def test_rejects_an_unknown_manage_action(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError, match="must be one of"):
            make_client(transport).manage_memory(id="m1", action="explode")  # type: ignore[arg-type]
        assert transport.requests == []

    def test_enforces_the_25_learning_ceiling(self) -> None:
        transport = FakeTransport()
        learnings = [{"kind": "semantic", "title": f"t{i}", "body": "b"} for i in range(26)]
        with pytest.raises(KiomonError, match="25"):
            make_client(transport, workspace_id="ws_1").reflect_session(learnings=learnings)
        assert transport.requests == []

    def test_accepts_exactly_25_learnings(self) -> None:
        transport = FakeTransport(lambda request: json_response(BATCH_OK))
        learnings = [{"kind": "semantic", "title": f"t{i}", "body": "b"} for i in range(25)]
        make_client(transport, workspace_id="ws_1").reflect_session(learnings=learnings)
        assert len(transport.requests) == 1

    def test_rejects_an_empty_learning_list(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError):
            make_client(transport).reflect_session(learnings=[])
        assert transport.requests == []

    @pytest.mark.parametrize(
        "learning",
        [
            {"kind": "opinion", "title": "t", "body": "b"},
            {"kind": "semantic", "title": " ", "body": "b"},
            {"kind": "semantic", "title": "t", "body": " "},
            {"kind": "semantic", "title": "t", "body": "b", "confidence": 1.5},
            {"kind": "semantic", "title": "t", "body": "b", "confidence": -0.1},
        ],
    )
    def test_rejects_invalid_learnings(self, learning: dict[str, Any]) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError):
            make_client(transport, workspace_id="ws_1").draft_memory(learning)  # type: ignore[arg-type]
        assert transport.requests == []

    def test_accepts_confidence_bounds(self) -> None:
        transport = FakeTransport(lambda request: json_response(BATCH_OK))
        client = make_client(transport, workspace_id="ws_1")
        client.draft_memory({"kind": "semantic", "title": "t", "body": "b", "confidence": 0})
        client.draft_memory({"kind": "semantic", "title": "t", "body": "b", "confidence": 1})
        assert len(transport.requests) == 2

    def test_enforces_the_retrieval_limit_ceiling(self) -> None:
        transport = FakeTransport()
        client = make_client(transport, workspace_id="ws_1")
        with pytest.raises(KiomonError, match="between 1 and 20"):
            client.retrieve_memory(intent="plan", query="q", limit=50)
        with pytest.raises(KiomonError, match="between 1 and 20"):
            client.retrieve_memory(intent="plan", query="q", limit=0)
        assert transport.requests == []

    def test_rejects_an_empty_retrieval_query(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError):
            make_client(transport).retrieve_memory(intent="plan", query=" ")
        assert transport.requests == []

    def test_rejects_empty_outcome_items(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError, match="non-empty"):
            make_client(transport).record_outcome(outcome="success", items=[])
        assert transport.requests == []

    def test_rejects_non_mapping_outcome_items(self) -> None:
        transport = FakeTransport()
        with pytest.raises(KiomonError):
            make_client(transport).record_outcome(outcome="success", items=["m1"])  # type: ignore[list-item]
        assert transport.requests == []

    def test_rejects_an_empty_workspace_selection(self) -> None:
        with pytest.raises(KiomonError):
            make_client(FakeTransport()).select_workspace("")
