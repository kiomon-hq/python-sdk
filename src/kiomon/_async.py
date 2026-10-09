"""Kiomon SDK — asynchronous client.

:class:`AsyncKiomon` is a native async client: it performs its own I/O over the
:class:`~kiomon._async_transport.AsyncioTransport` (asyncio streams, standard
library only), so awaiting it never blocks a thread and cancelling the await
cancels the socket work and closes the connection.

It shares :mod:`kiomon._specs` with the synchronous client — the same
validation, argument-to-request mapping, headers, retries and response parsing —
so the two surfaces cannot drift apart.

Argument validation runs inside the coroutine, so an invalid argument raises
:class:`~kiomon.errors.KiomonError` when the call is awaited (no request is
sent); the synchronous client raises before the call.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from ._async_transport import AsyncioTransport, AsyncTransport
from ._client import DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, TokenProvider
from ._specs import (
    Spec,
    backoff_seconds,
    batch_draft_spec,
    briefing_spec,
    build_headers,
    build_url,
    context_spec,
    get_memory_spec,
    graph_spec,
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
from .errors import KiomonError
from .types import (
    BatchDraftResult,
    CallOptions,
    ContextPacket,
    GraphResult,
    JsonObject,
    Learning,
    ManageAction,
    MemoryDocument,
    MemoryKind,
    Outcome,
    OutcomeItem,
    RetrievalIntent,
    RiskLevel,
    SearchResponse,
    WorkspaceContext,
    WorkspaceList,
)

__all__ = ["AsyncKiomon"]


async def _async_sleep(seconds: float) -> None:
    """Indirection for backoff waits, so tests never sleep for real."""
    await asyncio.sleep(seconds)


class AsyncKiomon:
    """Asynchronous client for the Kiomon API.

    Example:
        >>> async with AsyncKiomon(api_key="sk_kiomon_…") as kiomon:
        ...     packet = await kiomon.retrieve_memory(intent="plan", query="deployment")
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
        transport: AsyncTransport | None = None,
        headers: Mapping[str, str] | None = None,
        agent_id: str = "sdk-python",
    ) -> None:
        """
        Args:
            api_key: A Kiomon API key (``sk_kiomon_…``), sent as ``X-API-Key``.
            token: A delegated OAuth token (``koa_…``), sent as
                ``Authorization: Bearer``. Pass a callable — synchronous or
                ``async`` — to refresh tokens without rebuilding the client.
            workspace_id: Default workspace. Resolved automatically when the
                account has exactly one.
            base_url: Defaults to ``https://api.kiomon.com``.
            timeout: Per-attempt timeout in seconds. Defaults to 60.
            max_retries: Retries for retryable failures. Defaults to 2.
            transport: Injectable async transport (any object with
                ``async def send(request) -> Response``). Defaults to
                :class:`~kiomon._async_transport.AsyncioTransport`.
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
        self._transport = transport if transport is not None else AsyncioTransport()
        self._extra_headers = dict(headers or {})
        self._agent_id = agent_id

    def __repr__(self) -> str:
        auth = "api_key" if self._api_key else "token"
        return f"AsyncKiomon(base_url={self._base_url!r}, auth={auth!r}, agent_id={self._agent_id!r})"

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        """Release transport resources, if the transport owns any.

        :class:`~kiomon._async_transport.AsyncioTransport` owns none (each
        request owns its connection), so this is a no-op unless a custom
        transport exposes ``aclose()``.
        """
        closer = getattr(self._transport, "aclose", None)
        if callable(closer):
            await closer()

    async def __aenter__(self) -> AsyncKiomon:
        return self

    async def __aexit__(self, *exc_info: object) -> Literal[False]:
        await self.aclose()
        return False

    # ── Workspace selection ────────────────────────────────────────────────────

    @property
    def workspace_id(self) -> str | None:
        """The workspace calls default to, if one has been selected or resolved."""
        return self._workspace or self._resolved_workspace

    def select_workspace(self, workspace_id: str) -> AsyncKiomon:
        """Bind this client to a workspace. Returns the client so it can be chained."""
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise KiomonError("select_workspace requires a non-empty workspace id.", code="invalid_request")
        self._workspace = workspace_id
        return self

    # ── The 12 memory verbs ────────────────────────────────────────────────────

    async def search_memory(
        self,
        *,
        query: str,
        kind: MemoryKind | Sequence[MemoryKind] | None = None,
        workspace_id: str | None = None,
        limit: int | None = None,
        options: CallOptions | None = None,
    ) -> SearchResponse:
        """Hybrid keyword + semantic search over the library."""
        return await self._send(
            search_spec(
                query=query,
                kind=kind,
                workspace_id=self._workspace_for(workspace_id),
                limit=limit,
            ),
            options,
        )

    async def get_memory(self, id: str, *, options: CallOptions | None = None) -> MemoryDocument:
        """Fetch one memory by id."""
        return await self._send(get_memory_spec(id), options)

    async def get_workspace_context(
        self,
        *,
        workspace_id: str | None = None,
        topic: str | None = None,
        options: CallOptions | None = None,
    ) -> WorkspaceContext:
        """Orientation context for a workspace — the "where was I" pack."""
        resolved = await self._resolve_workspace(workspace_id)
        return await self._send(context_spec(workspace_id=resolved, topic=topic), options)

    async def get_latest_briefing(
        self,
        *,
        workspace_id: str | None = None,
        options: CallOptions | None = None,
    ) -> JsonObject:
        """The most recent proactive briefing for a workspace."""
        resolved = await self._resolve_workspace(workspace_id)
        return await self._send(briefing_spec(workspace_id=resolved), options)

    async def explore_memory_graph(
        self,
        *,
        id: str,
        relation: str | None = None,
        depth: int | None = None,
        options: CallOptions | None = None,
    ) -> GraphResult:
        """Multi-hop traversal of the memory graph from a centre node."""
        return await self._send(graph_spec(id=id, relation=relation, depth=depth), options)

    async def reflect_session(
        self,
        *,
        learnings: Sequence[Learning],
        workspace_id: str | None = None,
        options: CallOptions | None = None,
    ) -> BatchDraftResult:
        """Save durable learnings through the autonomous write gate (max 25)."""
        validated = validate_learnings(learnings)
        resolved = await self._resolve_workspace(workspace_id)
        return await self._send(batch_draft_spec(workspace_id=resolved, learnings=validated), options)

    async def draft_memory(
        self,
        learning: Learning,
        *,
        workspace_id: str | None = None,
        options: CallOptions | None = None,
    ) -> BatchDraftResult:
        """Save a single learning. Sugar over :meth:`reflect_session`."""
        validated = validate_learnings([learning])
        resolved = await self._resolve_workspace(workspace_id)
        return await self._send(batch_draft_spec(workspace_id=resolved, learnings=validated), options)

    async def manage_memory(
        self,
        *,
        id: str,
        action: ManageAction,
        value: int | None = None,
        options: CallOptions | None = None,
    ) -> JsonObject:
        """Lifecycle, feedback and review — one method for nine actions."""
        return await self._send(manage_spec(id=id, action=action, value=value), options)

    async def list_workspaces(self, *, options: CallOptions | None = None) -> WorkspaceList:
        """Every workspace this credential can reach."""
        return await self._send(Spec(method="GET", path="/api/workspaces"), options)

    async def retrieve_memory(
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
        """Intent-aware retrieval — the packet carries provenance and conflicts."""
        return await self._send(
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

    async def record_outcome(
        self,
        *,
        outcome: Outcome,
        items: Sequence[OutcomeItem],
        workspace_id: str | None = None,
        options: CallOptions | None = None,
    ) -> JsonObject:
        """Report whether retrieved memories actually helped."""
        return await self._send(
            record_outcome_spec(
                outcome=outcome,
                items=items,
                workspace_id=self._workspace_for(workspace_id),
            ),
            options,
        )

    # ── Escape hatch ───────────────────────────────────────────────────────────

    async def request(
        self,
        path: str,
        *,
        method: str = "GET",
        query: Mapping[str, Any] | None = None,
        body: Any = None,
        write: bool = False,
        options: CallOptions | None = None,
    ) -> Any:
        """Call any Kiomon REST route. Pass ``write=True`` for mutating routes."""
        return await self._send(Spec(method=method, path=path, query=query, body=body, write=write), options)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _workspace_for(self, explicit: str | None) -> str | None:
        """Synchronous workspace for query params, tolerant of an unresolved default."""
        return explicit or self._workspace or self._resolved_workspace

    async def _resolve_workspace(self, explicit: str | None) -> str:
        """Workspace for routes that need one in the path."""
        known = explicit or self._workspace or self._resolved_workspace
        if known:
            return known
        only = pick_workspace((await self.list_workspaces())["workspaces"])
        self._resolved_workspace = only
        return only

    async def _auth_headers(self) -> Mapping[str, str]:
        if self._api_key:
            return {"X-API-Key": self._api_key}
        token = self._token() if callable(self._token) else self._token
        if inspect.isawaitable(token):
            token = await token
        if not token:
            raise KiomonError("The token provider returned nothing.", code="unauthorized")
        return {"Authorization": f"Bearer {token}"}

    async def _send(self, spec: Spec, options: CallOptions | None = None) -> Any:
        url = build_url(self._base_url, spec.path, spec.query)
        headers = build_headers(
            agent_id=self._agent_id,
            extra_headers=self._extra_headers,
            auth=await self._auth_headers(),
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
                result = self._transport.send(
                    _Request(
                        method=spec.method,
                        url=url,
                        headers=headers,
                        body=payload,
                        timeout=self._timeout,
                    )
                )
                if not inspect.isawaitable(result):
                    raise KiomonError(
                        "The transport returned a response synchronously. `AsyncKiomon` needs an "
                        "async transport (an `async def send`); use `Kiomon` for a sync transport.",
                        code="invalid_request",
                    )
                response = await result
                return parse_response(response)
            except KiomonError as error:
                safe = not spec.write or bool(idempotency_key)
                if not error.retryable or not safe or attempt >= max_retries:
                    raise
                await _async_sleep(backoff_seconds(attempt, error))
                attempt += 1
