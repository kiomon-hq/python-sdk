"""Kiomon SDK — request planning.

The pure half of the client: validation, argument-to-request mapping, URL and
header assembly, response parsing and backoff arithmetic. Nothing here performs
I/O, and both :class:`~kiomon._client.Kiomon` and
:class:`~kiomon._async.AsyncKiomon` are built on it, so the synchronous and
asynchronous surfaces cannot drift apart.

Keeping the planning here also means the failure contract is shared: an invalid
argument raises :class:`~kiomon.errors.KiomonError` with ``code="invalid_request"``
before any transport is touched.
"""

from __future__ import annotations

import json
import random
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode

from ._transport import Response
from .errors import KiomonError, error_from_response
from .types import CallOptions

__all__ = [
    "LEARNING_KINDS",
    "MANAGE_ACTIONS",
    "MAX_LEARNINGS_PER_CALL",
    "MAX_RETRIEVAL_LIMIT",
    "Spec",
    "backoff_seconds",
    "batch_draft_spec",
    "briefing_spec",
    "build_headers",
    "build_url",
    "context_spec",
    "generate_key",
    "get_memory_spec",
    "graph_spec",
    "is_sequence",
    "kind_param",
    "manage_spec",
    "parse_response",
    "pick_workspace",
    "record_outcome_spec",
    "require_id",
    "retrieve_spec",
    "search_spec",
    "validate_client_options",
    "validate_learnings",
]

MAX_LEARNINGS_PER_CALL = 25
MAX_RETRIEVAL_LIMIT = 20

LEARNING_KINDS = ("semantic", "procedural", "episodic")
MANAGE_ACTIONS = ("pin", "unpin", "archive", "restore", "forget", "rate", "approve", "reject", "revert")


@dataclass(frozen=True)
class Spec:
    """One planned request, independent of how it will be executed."""

    method: str
    path: str
    query: Mapping[str, Any] | None = None
    body: Any = None
    #: Writes get an idempotency key and are retried only with one attached.
    write: bool = False


# ── Planners: arguments in, :class:`Spec` out ────────────────────────────────


def search_spec(
    *,
    query: str,
    kind: str | Sequence[str] | None,
    workspace_id: str | None,
    limit: int | None,
) -> Spec:
    if not isinstance(query, str) or not query.strip():
        raise KiomonError("search_memory requires a non-empty `query`.", code="invalid_request")
    return Spec(
        method="GET",
        path="/api/search",
        query={
            "q": query,
            "limit": limit if limit is not None else 10,
            "workspace_id": workspace_id,
            "kind": kind_param(kind),
        },
    )


def get_memory_spec(id: str) -> Spec:
    require_id(id, "get_memory")
    return Spec(method="GET", path=f"/api/documents/{_quote(id)}")


def context_spec(*, workspace_id: str, topic: str | None) -> Spec:
    return Spec(
        method="GET",
        path=f"/api/workspaces/{_quote(workspace_id)}/context",
        query={"topic": topic},
    )


def briefing_spec(*, workspace_id: str) -> Spec:
    return Spec(method="GET", path=f"/api/workspaces/{_quote(workspace_id)}/briefings/latest")


def graph_spec(*, id: str, relation: str | None, depth: int | None) -> Spec:
    require_id(id, "explore_memory_graph")
    return Spec(
        method="GET",
        path="/api/graph",
        query={"center_id": id, "depth": depth if depth is not None else 1, "relation": relation},
    )


def batch_draft_spec(*, workspace_id: str, learnings: Sequence[Mapping[str, Any]]) -> Spec:
    return Spec(
        method="POST",
        path="/api/memories/batch-draft",
        body={"workspace_id": workspace_id, "learnings": list(learnings)},
        write=True,
    )


def manage_spec(*, id: str, action: str, value: int | None) -> Spec:
    require_id(id, "manage_memory")
    if action not in MANAGE_ACTIONS:
        raise KiomonError(
            f"manage_memory: `action` must be one of {', '.join(MANAGE_ACTIONS)}.",
            code="invalid_request",
        )
    if action == "rate" and (isinstance(value, bool) or value not in (1, -1)):
        raise KiomonError('manage_memory: action "rate" requires `value` of 1 or -1.', code="invalid_request")

    encoded = _quote(id)
    if action in ("pin", "unpin"):
        return Spec(method="PATCH", path=f"/api/documents/{encoded}", body={"pinned": 1 if action == "pin" else 0}, write=True)
    if action == "rate":
        return Spec(method="POST", path=f"/api/documents/{encoded}/rate", body={"rating": value}, write=True)
    if action == "revert":
        return Spec(method="POST", path=f"/api/memories/{encoded}/revert", write=True)
    return Spec(method="POST", path=f"/api/documents/{encoded}/{action}", write=True)


def retrieve_spec(
    *,
    intent: str,
    query: str,
    entities: Sequence[str] | None,
    constraints: Sequence[str] | None,
    kinds: Sequence[str] | None,
    workspace_id: str | None,
    risk_level: str | None,
    limit: int | None,
) -> Spec:
    if not isinstance(query, str) or not query.strip():
        raise KiomonError("retrieve_memory requires a non-empty `query`.", code="invalid_request")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_RETRIEVAL_LIMIT):
        raise KiomonError(
            f"retrieve_memory: `limit` must be between 1 and {MAX_RETRIEVAL_LIMIT}.",
            code="invalid_request",
        )

    body: dict = {"intent": intent, "query": query}
    if entities is not None:
        body["entities"] = list(entities)
    if constraints is not None:
        body["constraints"] = list(constraints)
    if kinds is not None:
        body["kinds"] = list(kinds)
    if risk_level is not None:
        body["risk_level"] = risk_level
    if limit is not None:
        body["limit"] = limit
    if workspace_id is not None:
        body["workspace_id"] = workspace_id

    # ``write=False`` on purpose. retrieve is a POST that only reads; keying it
    # would have the server store one row per retrieval that no retry could
    # ever reuse.
    return Spec(method="POST", path="/api/retrieve", body=body)


def record_outcome_spec(
    *,
    outcome: str,
    items: Sequence[Mapping[str, Any]],
    workspace_id: str | None,
) -> Spec:
    if not is_sequence(items) or len(items) == 0:
        raise KiomonError("record_outcome requires a non-empty `items` sequence.", code="invalid_request")
    normalized = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise KiomonError(f"record_outcome: items[{index}] must be a mapping.", code="invalid_request")
        normalized.append(dict(item))

    body: dict = {"outcome": outcome, "items": normalized}
    if workspace_id is not None:
        body["workspace_id"] = workspace_id
    return Spec(method="POST", path="/api/memories/record-outcome", body=body, write=True)


# ── Client options ─────────────────────────────────────────────────────────


def validate_client_options(*, api_key: Any, token: Any, timeout: Any, max_retries: Any) -> None:
    """Validate constructor arguments before a client is built."""
    if not api_key and not token:
        raise KiomonError(
            "Kiomon was constructed without credentials. Pass `api_key` (server-side) "
            "or `token` (delegated OAuth).",
            code="invalid_request",
        )
    if api_key and token:
        raise KiomonError(
            "Pass either `api_key` or `token`, not both — the wire format differs and "
            "only one can be sent.",
            code="invalid_request",
        )
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise KiomonError("`timeout` must be a positive number of seconds.", code="invalid_request")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise KiomonError("`max_retries` must be a non-negative integer.", code="invalid_request")


# ── Validation ───────────────────────────────────────────────────────────────


def is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def require_id(id: Any, method: str) -> None:
    if not isinstance(id, str) or not id.strip():
        raise KiomonError(f"{method} requires a non-empty memory id.", code="invalid_request")


def kind_param(kind: str | Sequence[str] | None) -> str | None:
    """Accept one kind or several; the API takes a comma-separated string."""
    if kind is None:
        return None
    if isinstance(kind, str):
        return kind
    if is_sequence(kind):
        return ",".join(str(item) for item in kind)
    raise KiomonError("search_memory: `kind` must be a string or a sequence of strings.", code="invalid_request")


def validate_learnings(learnings: Any) -> list[Mapping[str, Any]]:
    """Mirror the worker-side validation the MCP tools perform, so failures are local and cheap."""
    if not is_sequence(learnings) or len(learnings) == 0:
        raise KiomonError("`learnings` must be a non-empty sequence.", code="invalid_request")
    if len(learnings) > MAX_LEARNINGS_PER_CALL:
        raise KiomonError(
            f"`learnings` cannot exceed {MAX_LEARNINGS_PER_CALL} items per call.",
            code="invalid_request",
        )

    validated: list[Mapping[str, Any]] = []
    for index, learning in enumerate(learnings):
        if not isinstance(learning, Mapping):
            raise KiomonError(f"learnings[{index}] must be a mapping.", code="invalid_request")
        if learning.get("kind") not in LEARNING_KINDS:
            raise KiomonError(
                f"learnings[{index}].kind must be one of: {', '.join(LEARNING_KINDS)}.",
                code="invalid_request",
            )
        title = learning.get("title")
        if not isinstance(title, str) or not title.strip():
            raise KiomonError(f"learnings[{index}].title is required.", code="invalid_request")
        body = learning.get("body")
        if not isinstance(body, str) or not body.strip():
            raise KiomonError(f"learnings[{index}].body is required.", code="invalid_request")
        confidence = learning.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1
        ):
            raise KiomonError(
                f"learnings[{index}].confidence must be between 0 and 1.",
                code="invalid_request",
            )
        # ``None`` means "not set" in Python; dropping the key keeps the wire
        # identical to the TypeScript SDK's ``undefined`` handling.
        validated.append({key: value for key, value in learning.items() if value is not None})
    return validated


# ── Workspace resolution ─────────────────────────────────────────────────────


def pick_workspace(workspaces: Sequence[Mapping[str, Any]]) -> str:
    """Resolve the only workspace, or explain why there is no safe default.

    Exactly one workspace resolves silently (matching the MCP behaviour); more
    than one is an error the caller must fix, because guessing which brain to
    write to would be worse than failing.
    """
    if len(workspaces) == 0:
        raise KiomonError(
            "This account has no workspaces yet. Create one in the Kiomon dashboard.",
            code="not_found",
        )
    if len(workspaces) > 1:
        names = ", ".join(f"{w.get('slug')} ({w.get('id')})" for w in workspaces)
        raise KiomonError(
            f"This account has {len(workspaces)} workspaces, so there is no safe default. "
            f"Pass `workspace_id` to the constructor, or call select_workspace(). Available: {names}",
            code="invalid_request",
        )
    return str(workspaces[0]["id"])


# ── URL, headers, response ───────────────────────────────────────────────────


def build_url(base_url: str, path: str, query: Mapping[str, Any] | None) -> str:
    if not path.startswith("/"):
        path = f"/{path}"
    url = f"{base_url}{path}"
    if query:
        pairs = [(key, value) for key, value in query.items() if value is not None]
        if pairs:
            url = f"{url}?{urlencode(pairs)}"
    return url


def build_headers(
    *,
    agent_id: str,
    extra_headers: Mapping[str, str],
    auth: Mapping[str, str],
    body: Any,
    write: bool,
    options: CallOptions | None,
) -> dict:
    headers: dict = {"Accept": "application/json", "X-Agent-Id": agent_id}
    headers.update(extra_headers)
    headers.update(auth)
    if body is not None:
        headers["Content-Type"] = "application/json"
    if write:
        # A write is only safely retryable with a stable key, so one is always
        # attached — generated if the caller did not supply one — and the same
        # key is reused on every attempt.
        headers["Idempotency-Key"] = (
            options.idempotency_key if options is not None and options.idempotency_key else generate_key()
        )
    if options is not None and options.request_id:
        headers["X-Request-Id"] = options.request_id
    return headers


def parse_response(response: Response) -> Any:
    if 200 <= response.status < 300:
        if response.status == 204 or not response.body:
            return None
        try:
            return json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise KiomonError(
                f"Kiomon returned a non-JSON response (HTTP {response.status}).",
                code="service_unavailable",
                status=response.status,
                cause=exc,
            ) from exc
    raise error_from_response(status=response.status, body=response.body, headers=response.headers)


# ── Retry arithmetic ─────────────────────────────────────────────────────────


def generate_key() -> str:
    return str(uuid.uuid4())


def backoff_seconds(attempt: int, error: KiomonError) -> float:
    """Exponential backoff with jitter, honouring ``retry_after`` when sent."""
    if error.retry_after is not None and error.retry_after > 0:
        return min(error.retry_after, 30.0)
    base = min(0.25 * 2**attempt, 4.0)
    return base + random.uniform(0, 0.1)


def _quote(value: str) -> str:
    return quote(value, safe="")
