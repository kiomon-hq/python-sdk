"""Kiomon SDK — synchronous client.

The public vocabulary is the 12 memory-native verbs, mirroring Kiomon's hosted
MCP toolset method for method (the Python names are the snake_case MCP tool
names), so a developer can move between Python, TypeScript and an agent without
relearning anything. Every verb maps onto an existing REST endpoint; the SDK
adds typing, retries, idempotency and stable error codes.

Four further verbs — :meth:`~Kiomon.create_workspace`,
:meth:`~Kiomon.ingest_document`, :meth:`~Kiomon.list_documents` and
:meth:`~Kiomon.delete_document` — exist because the memory verbs alone are not
self-sufficient: without them a caller cannot make the workspace that
:meth:`~Kiomon.select_workspace` needs, or put anything in it for
:meth:`~Kiomon.search_memory` to find. They are SDK-only; the MCP toolset has no
equivalent.

Request planning — validation, body construction, headers, response parsing and
backoff — lives in :mod:`kiomon._specs` and is shared verbatim with the
async client, so the two surfaces cannot drift apart.

Failure contract: an invalid *argument* raises synchronously, before any network
call, so a programming error is never buried behind I/O. Everything that can
fail for *remote* reasons — auth, quota, transport, a 5xx — raises
:class:`~kiomon.errors.KiomonError`.

Local validation is deliberately the same set the TypeScript SDK validates
(constructor credentials, ids and queries, the 25-learning ceiling, the
0–1 confidence range, the manage-action vocabulary, the retrieval and document
page ceilings). Everything else is the server's contract and is validated there.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

from ._specs import (
    Spec,
    backoff_seconds,
    batch_draft_spec,
    briefing_spec,
    build_headers,
    build_url,
    context_spec,
    create_workspace_spec,
    delete_document_spec,
    get_memory_spec,
    graph_spec,
    ingest_document_spec,
    list_documents_spec,
    list_memories_spec,
    manage_spec,
    parse_response,
    pick_workspace,
    record_outcome_spec,
    retrieve_spec,
    search_spec,
    validate_client_options,
    validate_learnings,
)
from ._transport import Request as _Request
from ._transport import Transport, UrllibTransport
from .errors import KiomonError
from .types import (
    BatchDraftResult,
    CallOptions,
    ContextPacket,
    DeleteResult,
    DocumentList,
    DocumentSort,
    DocumentStatus,
    GraphResult,
    IndexStatus,
    IngestResult,
    JsonObject,
    Learning,
    ManageAction,
    MemoryDocument,
    MemoryKind,
    MemoryList,
    MemoryStatus,
    Outcome,
    OutcomeItem,
    RetrievalIntent,
    RiskLevel,
    SearchResponse,
    Workspace,
    WorkspaceContext,
    WorkspaceList,
)

__all__ = ["DEFAULT_BASE_URL", "DEFAULT_MAX_RETRIES", "DEFAULT_TIMEOUT", "Kiomon", "TokenProvider"]

#: Where the hosted API lives, unless ``base_url`` overrides it.
DEFAULT_BASE_URL = "https://api.kiomon.com"
#: Per-attempt timeout, in seconds.
DEFAULT_TIMEOUT = 60.0
#: Retries for retryable failures (so up to 3 attempts by default).
DEFAULT_MAX_RETRIES = 2

#: A token, or a callable resolved once per request so refresh needs no rebuild.
#: :class:`~kiomon._async.AsyncKiomon` also accepts an async callable.
TokenProvider = str | Callable[[], str | None] | Callable[[], Awaitable[str | None]]


def _sleep(seconds: float) -> None:
    """Indirection for backoff waits, so tests never sleep for real."""
    time.sleep(seconds)


class Kiomon:
    """Synchronous client for the Kiomon API.

    Example:
        >>> kiomon = Kiomon(api_key="sk_kiomon_…")
        >>> packet = kiomon.retrieve_memory(intent="execute", query="how do we deploy?")
        >>> kiomon.record_outcome(
        ...     outcome="success",
        ...     items=[{"memory_id": mid, "assessment": "helpful"} for mid in packet["retrieved_ids"]],
        ... )
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        token: TokenProvider | None = None,
        workspace_id: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        transport: Transport | None = None,
        headers: Mapping[str, str] | None = None,
        agent_id: str = "sdk-python",
    ) -> None:
        """
        Args:
            api_key: A Kiomon API key (``sk_kiomon_…``), sent as ``X-API-Key``.
                Use for server-side integrations.
            token: A delegated OAuth token (``koa_…``), sent as
                ``Authorization: Bearer``. Pass a callable to refresh tokens
                without rebuilding the client. A connector token is confined to
                the memory routes by the worker's default-deny allowlist, so
                :meth:`request` is not a general escape hatch in this mode.
            workspace_id: Default workspace. Resolved automatically when the
                account has exactly one.
            base_url: Defaults to ``https://api.kiomon.com``.
            timeout: Per-attempt timeout in seconds. Defaults to 60.
            max_retries: Retries for retryable failures. Defaults to 2.
            transport: Injectable HTTP transport for tests and non-standard
                runtimes. Defaults to :class:`~kiomon._transport.UrllibTransport`.
            headers: Extra headers on every request.
            agent_id: Attributed in Kiomon's telemetry as ``X-Agent-Id``.
        """
        validate_client_options(api_key=api_key, token=token, timeout=timeout, max_retries=max_retries)

        self._api_key = api_key
        self._token = token
        self._workspace = workspace_id
        self._resolved_workspace: str | None = None
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._timeout = float(timeout)
        self._max_retries = max_retries
        self._transport = transport if transport is not None else UrllibTransport()
        self._extra_headers = dict(headers or {})
        self._agent_id = agent_id

    def __repr__(self) -> str:
        auth = "api_key" if self._api_key else "token"
        return f"Kiomon(base_url={self._base_url!r}, auth={auth!r}, agent_id={self._agent_id!r})"

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release transport resources, if the transport owns any.

        The default :class:`~kiomon._transport.UrllibTransport` owns none, so
        this is a no-op unless a custom transport exposes ``close()``.
        """
        closer = getattr(self._transport, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> Kiomon:
        return self

    def __exit__(self, *exc_info: object) -> Literal[False]:
        self.close()
        return False

    # ── Workspace selection ────────────────────────────────────────────────────

    @property
    def workspace_id(self) -> str | None:
        """The workspace calls default to, if one has been selected or resolved."""
        return self._workspace or self._resolved_workspace

    def select_workspace(self, workspace_id: str) -> Kiomon:
        """Bind this client to a workspace, mirroring MCP ``select_workspace``.

        Returns the client so it can be chained::

            kiomon = Kiomon(api_key=key).select_workspace("ws_…")
        """
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise KiomonError("select_workspace requires a non-empty workspace id.", code="invalid_request")
        self._workspace = workspace_id
        return self

    # ── The 12 memory verbs ────────────────────────────────────────────────────

    def search_memory(
        self,
        *,
        query: str,
        kind: MemoryKind | Sequence[MemoryKind] | None = None,
        workspace_id: str | None = None,
        limit: int | None = None,
        options: CallOptions | None = None,
    ) -> SearchResponse:
        """Hybrid keyword + semantic search over the library."""
        return self._send(
            search_spec(
                query=query,
                kind=kind,
                workspace_id=self._workspace_for(workspace_id),
                limit=limit,
            ),
            options,
        )

    def get_memory(self, id: str, *, options: CallOptions | None = None) -> MemoryDocument:
        """Fetch one memory by id."""
        return self._send(get_memory_spec(id), options)

    def get_workspace_context(
        self,
        *,
        workspace_id: str | None = None,
        topic: str | None = None,
        options: CallOptions | None = None,
    ) -> WorkspaceContext:
        """Orientation context for a workspace — the "where was I" pack."""
        resolved = self._resolve_workspace(workspace_id)
        return self._send(context_spec(workspace_id=resolved, topic=topic), options)

    def get_latest_briefing(
        self,
        *,
        workspace_id: str | None = None,
        options: CallOptions | None = None,
    ) -> JsonObject:
        """The most recent proactive briefing for a workspace."""
        resolved = self._resolve_workspace(workspace_id)
        return self._send(briefing_spec(workspace_id=resolved), options)

    def explore_memory_graph(
        self,
        *,
        id: str,
        relation: str | None = None,
        depth: int | None = None,
        workspace_id: str | None = None,
        options: CallOptions | None = None,
    ) -> GraphResult:
        """Multi-hop traversal of the memory graph from a centre node."""
        return self._send(
            graph_spec(
                id=id,
                relation=relation,
                depth=depth,
                workspace_id=self._workspace_for(workspace_id),
            ),
            options,
        )

    def reflect_session(
        self,
        *,
        learnings: Sequence[Learning],
        workspace_id: str | None = None,
        options: CallOptions | None = None,
    ) -> BatchDraftResult:
        """Save durable learnings from a session through the autonomous write gate.

        High-confidence learnings may activate immediately; the rest land in the
        Review Inbox. Maximum 25 learnings per call.
        """
        validated = validate_learnings(learnings)
        resolved = self._resolve_workspace(workspace_id)
        return self._send(batch_draft_spec(workspace_id=resolved, learnings=validated), options)

    def draft_memory(
        self,
        learning: Learning,
        *,
        workspace_id: str | None = None,
        options: CallOptions | None = None,
    ) -> BatchDraftResult:
        """Save a single learning. Sugar over :meth:`reflect_session`."""
        validated = validate_learnings([learning])
        resolved = self._resolve_workspace(workspace_id)
        return self._send(batch_draft_spec(workspace_id=resolved, learnings=validated), options)

    def manage_memory(
        self,
        *,
        id: str,
        action: ManageAction,
        value: int | None = None,
        options: CallOptions | None = None,
    ) -> JsonObject:
        """Lifecycle, feedback and review — one method, mirroring MCP ``manage_memory``.

        Actions: ``pin`` · ``unpin`` · ``archive`` · ``restore`` · ``forget`` ·
        ``rate`` (``value`` 1 or -1) · ``approve`` · ``reject`` · ``revert``.
        """
        return self._send(manage_spec(id=id, action=action, value=value), options)

    def list_workspaces(self, *, options: CallOptions | None = None) -> WorkspaceList:
        """Every workspace this credential can reach."""
        return self._send(Spec(method="GET", path="/api/workspaces"), options)

    # ── Workspace and document verbs ───────────────────────────────────────────
    #
    # SDK-only: the MCP toolset has no equivalent, and without these the memory
    # verbs cannot be used from a cold start — a caller could not create the
    # workspace `select_workspace` needs, nor put anything in it for
    # `search_memory` to find.

    def create_workspace(
        self,
        *,
        name: str,
        org_id: str | None = None,
        options: CallOptions | None = None,
    ) -> Workspace:
        """Create a workspace, optionally inside an organization.

        Omit ``org_id`` to file it under the caller's primary organization; pass
        one to place it in a specific org, where the caller must be an admin or
        owner or the API answers 403. Returns the bare workspace row — the
        enriched fields (``documents``, ``my_role``, ``is_default``) come from
        :meth:`list_workspaces`, not from this call.

        The plan's workspace ceiling applies: a Free organization over its limit
        gets 402 ``quota_exceeded``, not a silent no-op.
        """
        return self._send(create_workspace_spec(name=name, org_id=org_id), options)

    def ingest_document(
        self,
        *,
        body: str,
        title: str | None = None,
        source_url: str | None = None,
        author: str | None = None,
        excerpt: str | None = None,
        lang: str | None = None,
        user_notes: str | None = None,
        tags: Sequence[str] | None = None,
        kind: MemoryKind | None = None,
        type: str | None = None,
        thumbnail_url: str | None = None,
        workspace_id: str | None = None,
        options: CallOptions | None = None,
    ) -> IngestResult:
        """Save a document — the write path for source material.

        Distinct from :meth:`draft_memory`: a learning goes through the write
        gate and may land in the Review Inbox, while a document is stored and
        indexed directly. Documents are what :meth:`search_memory` returns.

        Indexing is asynchronous, so the row is not searchable when this
        returns. Expect a lag — seconds, not instant — before the id shows up in
        a search.

        Dedup is by ``source_url`` **only**. Re-saving the same URL answers with
        ``duplicate: True`` and the **existing** id rather than an error — but a
        save with no ``source_url`` is given a synthetic unique one, so it never
        dedups and repeats will pile up as separate documents. Pass ``source_url``
        if you want the second save to be a no-op.
        """
        return self._send(
            ingest_document_spec(
                body=body,
                title=title,
                source_url=source_url,
                author=author,
                excerpt=excerpt,
                lang=lang,
                user_notes=user_notes,
                tags=tags,
                kind=kind,
                type=type,
                thumbnail_url=thumbnail_url,
                workspace_id=self._workspace_for(workspace_id),
            ),
            options,
        )

    def list_documents(
        self,
        *,
        query: str | None = None,
        kind: MemoryKind | None = None,
        tag: str | None = None,
        status: DocumentStatus | None = None,
        source: str | None = None,
        sort: DocumentSort | None = None,
        since: str | None = None,
        index_status: IndexStatus | None = None,
        limit: int | None = None,
        offset: int | None = None,
        workspace_id: str | None = None,
        options: CallOptions | None = None,
    ) -> DocumentList:
        """List documents, newest first by default.

        Metadata only — the body is not in the projection, so read it with
        :meth:`get_memory` when you need it. ``counts`` reports the per-kind
        totals for the current filter over the whole match set, not just the
        page.

        **Only indexed documents are listed by default.** ``queued`` and
        ``failed`` rows are hidden, so a document you just saved is absent until
        it finishes chunking — pass ``index_status="all"`` to see the pipeline
        rather than just its output. That is also the way to find a save that
        failed: it is invisible to the default filter forever.
        """
        return self._send(
            list_documents_spec(
                limit=limit,
                offset=offset,
                query=query,
                kind=kind,
                tag=tag,
                status=status,
                source=source,
                sort=sort,
                since=since,
                index_status=index_status,
                workspace_id=self._workspace_for(workspace_id),
            ),
            options,
        )

    def delete_document(
        self,
        id: str,
        *,
        options: CallOptions | None = None,
    ) -> DeleteResult:
        """Delete a document and everything derived from it.

        The row, its blob, its chunks, tags, links and both chunk- and
        document-level vectors. Irreversible — prefer ``archive`` via
        :meth:`manage_memory` when the intent is to hide rather than remove.
        """
        return self._send(delete_document_spec(id), options)

    def list_memories(
        self,
        *,
        status: MemoryStatus | None = None,
        kind: MemoryKind | None = None,
        query: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        workspace_id: str | None = None,
        options: CallOptions | None = None,
    ) -> MemoryList:
        """List memories and files by state — the listing a query cannot express.

        ``status`` defaults to ``"pending_approval"``, the **Review Inbox**:
        drafts the write gate held back for a decision. They are not retrievable
        — nothing is until it is approved — so a search that returns nothing is
        not proof the brain is empty. Use ``"active"`` for live knowledge,
        ``"archived"`` for what was hidden or quarantined, ``"all"`` for
        everything.

        Work the inbox with
        :meth:`manage_memory` (``action="approve"`` or ``"reject"``).
        """
        return self._send(
            list_memories_spec(
                status=status,
                kind=kind,
                query=query,
                limit=limit,
                offset=offset,
                workspace_id=self._workspace_for(workspace_id),
            ),
            options,
        )

    def retrieve_memory(
        self,
        *,
        intent: RetrievalIntent,
        query: str,
        entities: Sequence[str] | None = None,
        constraints: Sequence[str] | None = None,
        kinds: Sequence[MemoryKind] | None = None,
        workspace_id: str | None = None,
        risk_level: RiskLevel | None = None,
        limit: int | None = None,
        options: CallOptions | None = None,
    ) -> ContextPacket:
        """Intent-aware retrieval.

        Prefer this over :meth:`search_memory` when feeding a model: the packet
        carries provenance, confidence and explicit conflict warnings.
        """
        return self._send(
            retrieve_spec(
                intent=intent,
                query=query,
                entities=entities,
                constraints=constraints,
                kinds=kinds,
                workspace_id=self._workspace_for(workspace_id),
                risk_level=risk_level,
                limit=limit,
            ),
            options,
        )

    def record_outcome(
        self,
        *,
        outcome: Outcome,
        items: Sequence[OutcomeItem],
        workspace_id: str | None = None,
        options: CallOptions | None = None,
    ) -> JsonObject:
        """Report whether retrieved memories actually helped.

        This is what lets the brain re-rank and decay honestly, so call it after
        acting on a packet.
        """
        return self._send(
            record_outcome_spec(
                outcome=outcome,
                items=items,
                workspace_id=self._workspace_for(workspace_id),
            ),
            options,
        )

    # ── Escape hatch ───────────────────────────────────────────────────────────

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        query: Mapping[str, Any] | None = None,
        body: Any = None,
        write: bool = False,
        options: CallOptions | None = None,
    ) -> Any:
        """Call any Kiomon REST route.

        Exists so the SDK never becomes a blocker for a new endpoint. Pass
        ``write=True`` for a route that mutates state, so it gets an
        ``Idempotency-Key`` and is retried safely. Unavailable in delegated-token
        mode for routes outside the memory surface — the worker's allowlist
        rejects them.
        """
        return self._send(Spec(method=method, path=path, query=query, body=body, write=write), options)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _workspace_for(self, explicit: str | None) -> str | None:
        """Synchronous workspace for query params, tolerant of an unresolved default."""
        return explicit or self._workspace or self._resolved_workspace

    def _resolve_workspace(self, explicit: str | None) -> str:
        """Workspace for routes that need one in the path."""
        known = explicit or self._workspace or self._resolved_workspace
        if known:
            return known
        only = pick_workspace(self.list_workspaces()["workspaces"])
        self._resolved_workspace = only
        return only
    def _auth_headers(self) -> Mapping[str, str]:
        if self._api_key:
            return {"X-API-Key": self._api_key}
        token = self._token() if callable(self._token) else self._token
        if inspect.isawaitable(token):
            _close_awaitable(token)
            raise KiomonError(
                "The token provider returned an awaitable. `Kiomon` resolves tokens synchronously; "
                "use `AsyncKiomon` for an async token provider.",
                code="invalid_request",
            )
        if not token:
            raise KiomonError("The token provider returned nothing.", code="unauthorized")
        return {"Authorization": f"Bearer {token}"}

    def _send(self, spec: Spec, options: CallOptions | None = None) -> Any:
        url = build_url(self._base_url, spec.path, spec.query)
        headers = build_headers(
            agent_id=self._agent_id,
            extra_headers=self._extra_headers,
            auth=self._auth_headers(),
            body=spec.body,
            write=spec.write,
            options=options,
        )
        idempotency_key = headers.get("Idempotency-Key")
        max_retries = self._max_retries if options is None or options.max_retries is None else max(0, options.max_retries)
        payload = None if spec.body is None else json.dumps(spec.body).encode("utf-8")

        attempt = 0
        while True:
            try:
                response = self._transport.send(
                    _Request(
                        method=spec.method,
                        url=url,
                        headers=headers,
                        body=payload,
                        timeout=self._timeout,
                    )
                )
                if inspect.isawaitable(response):
                    _close_awaitable(response)
                    raise KiomonError(
                        "The transport returned an awaitable. `Kiomon` needs a synchronous transport; "
                        "use `AsyncKiomon` with an async transport.",
                        code="invalid_request",
                    )
                return parse_response(response)
            except KiomonError as error:
                safe = not spec.write or bool(idempotency_key)
                if not error.retryable or not safe or attempt >= max_retries:
                    raise
                _sleep(backoff_seconds(attempt, error))
                attempt += 1


def _close_awaitable(value: Any) -> None:
    """Dispose of a coroutine we refuse to await, so Python does not warn."""
    closer = getattr(value, "close", None)
    if callable(closer):
        closer()
