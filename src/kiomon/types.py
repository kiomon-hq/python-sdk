"""Kiomon SDK — public types.

Field names mirror the wire format exactly (``workspace_id``, ``memory_id``,
``source_refs``); the one deliberate exception is ``GraphResult.entityLinks``,
which is camelCase on the wire. Renaming fields in the SDK would break the
ability to diff a Python call against the equivalent TypeScript or MCP call,
which is the whole point of keeping the surfaces identical.

These TypedDicts are the SDK's canonical public contract: responses are plain
dictionaries, so nothing is lost in translation and no runtime model library is
needed. Optional keys are declared with the ``total=False`` subclass pattern so
the package stays importable on Python 3.10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

__all__ = [
    "Applicability",
    "Assessment",
    "BatchDraftMemory",
    "BatchDraftResult",
    "CallOptions",
    "ConflictNote",
    "ContextPacket",
    "DeleteResult",
    "DocumentCounts",
    "DocumentListItem",
    "DocumentList",
    "DocumentSort",
    "DocumentStatus",
    "EvidenceRef",
    "GraphRelation",
    "GraphResult",
    "IndexStatus",
    "IngestResult",
    "JsonObject",
    "Learning",
    "LearningKind",
    "ManageAction",
    "MemoryDocument",
    "MemoryKind",
    "MemoryList",
    "MemoryStatus",
    "MemorySummary",
    "Outcome",
    "OutcomeItem",
    "PacketItem",
    "RetrievalIntent",
    "RiskLevel",
    "SearchHit",
    "SearchResponse",
    "Workspace",
    "WorkspaceContext",
    "WorkspaceList",
]

# ── Enumerations (string literals, mirroring the wire values) ─────────────────

#: Memory kind. ``semantic`` = a fact, ``procedural`` = a routine,
#: ``episodic`` = a moment, ``reference`` = saved source material.
MemoryKind = Literal["reference", "episodic", "semantic", "procedural"]

#: Document lifecycle state, for :meth:`Kiomon.list_documents`.
DocumentStatus = Literal["all", "active", "archived", "pending_approval"]

#: Indexing state, for :meth:`Kiomon.list_documents`.
#:
#: ``chunked``, ``indexed`` and ``ready`` are **findable** — search can see them.
#: ``queued`` and ``failed`` are not, and ``list_documents`` hides them unless
#: ``index_status="all"`` is asked for explicitly. That default is why a document
#: you just ingested is missing from the list until it finishes chunking.
IndexStatus = Literal["all", "queued", "chunked", "indexed", "ready", "failed"]

#: Page ordering, for :meth:`Kiomon.list_documents`.
DocumentSort = Literal["recent", "oldest", "recently_read"]

#: The kinds a learning can be saved as (``reference`` is ingested, not written).
LearningKind = Literal["semantic", "procedural", "episodic"]

#: The intent a retrieval is serving. Sharper intents return sharper packets.
RetrievalIntent = Literal["execute", "plan", "answer", "verify", "debug"]

RiskLevel = Literal["low", "medium", "high"]

#: How a retrieved memory performed, for :meth:`Kiomon.record_outcome`.
Assessment = Literal["helpful", "unused", "misleading", "outdated"]

Outcome = Literal["success", "failure"]

ManageAction = Literal[
    "pin",
    "unpin",
    "archive",
    "restore",
    "forget",
    "rate",
    "approve",
    "reject",
    "revert",
]

GraphRelation = Literal[
    "derived_from",
    "supports",
    "contradicts",
    "related_to",
    "precedes",
    "follows",
    "part_of",
]

#: How confident the server is that a packet item applies to the caller's task.
Applicability = Literal["applicable", "conditional", "unknown"]

#: Any object shape the SDK does not model precisely.
JsonObject = dict[str, Any]


# ── Responses ─────────────────────────────────────────────────────────────────


class _WorkspaceRequired(TypedDict):
    id: str
    user_id: str
    name: str
    slug: str
    created_at: str


class Workspace(_WorkspaceRequired, total=False):
    """A workspace.

    The list route enriches the row; the create route does not. ``documents``,
    ``my_role`` and ``is_default`` are the fields a caller needs to choose a
    workspace, and they are present only on :meth:`Kiomon.list_workspaces`
    responses — :meth:`Kiomon.create_workspace` returns the bare row.
    """

    org_id: str | None
    #: ``autonomous`` or ``review`` — whether drafts activate or wait in the inbox.
    review_mode: str
    my_role: str
    org_name: str | None
    org_role: str | None
    members: int
    documents: int
    storage_bytes: int
    #: Human-readable size (``"1.68 KB"``), the display twin of ``storage_bytes``.
    storage: str
    is_default: bool


class WorkspaceList(TypedDict):
    workspaces: list[Workspace]


class _MemoryDocumentRequired(TypedDict):
    id: str
    source_url: str
    title: str | None
    author: str | None
    excerpt: str | None
    tags: str | None
    user_notes: str | None
    body: str | None
    pinned: int
    created_at: str
    workspace_id: str
    source: str
    kind: MemoryKind
    confidence: float | None
    corroboration_count: int
    importance: float
    strength: float
    retrieval_count: int
    last_retrieved_at: str | None
    feedback_score: float
    feedback_count: int
    scope: str
    status: str
    expires_at: str | None


class MemoryDocument(_MemoryDocumentRequired, total=False):
    """A stored memory/document. Mirrors the worker's ``Document`` contract.

    ``tags``, ``headings`` and ``memory_metadata`` arrive as JSON **strings**,
    not arrays or objects — ``json.loads`` them to use them.

    The row is the server's, not the SDK's, so the fields below are declared
    optional: a given document carries whichever of them its lifecycle has
    reached. ``index_status`` / ``ingest_status`` / ``enrich_status`` are the
    processing states, and ``index_ready_at`` is the one that answers "is this
    findable yet?" — ``strength`` and ``retrieval_count`` are the memory-decay
    signals.
    """

    #: JSON array string, like ``tags``.
    headings: str | None
    #: JSON object string.
    memory_metadata: str | None
    user_id: str
    updated_at: str | None
    lang: str | None
    type: str | None
    thumbnail_url: str | None
    is_large: int
    #: Set when the body lives in blob storage rather than the row.
    r2_key: str | None
    byte_size: int
    occurred_at: str | None
    pending_suggestion_count: int
    caller_role: str
    #: Indexing: ``queued`` · ``chunked`` · ``indexed`` · ``ready`` · ``failed``.
    #: Only ``chunked``/``indexed``/``ready`` are findable — see :data:`IndexStatus`.
    index_status: str
    index_error: str | None
    index_attempts: int
    index_changed_at: str | None
    index_ready_at: str | None
    #: Ingest (the phase before indexing), same vocabulary.
    ingest_status: str
    ingest_error: str | None
    ingest_attempts: int
    ingest_changed_at: str | None
    ingest_pages: int | None
    ingest_parser_name: str | None
    ingest_parser_version: str | None
    #: Enrichment: extraction of entities, claims and domains.
    enrich_status: str
    enrich_error: str | None
    failed_at: str | None
    failure_class: str | None
    infra_retries: int
    first_stranded_at: str | None
    last_progress_at: str | None
    last_read_at: str | None
    #: Recovery-lease bookkeeping, written by the sweep jobs.
    lease_stage: str | None
    lease_until: str | None
    chunk_strategy_version: int
    embed_model: str | None
    attention_reason: str | None
    next_attention_at: str | None
    #: Auto-linking state and degree.
    link_status: str | None
    link_count: int
    last_link_scan_at: str | None
    connector_id: str | None
    connector_metadata: str | None


class _SearchHitRequired(TypedDict):
    id: str
    title: str | None
    source_url: str
    excerpt: str | None
    #: Query-matched fragment from the document body.
    snippet: str
    tags: str | None
    created_at: str


class SearchHit(_SearchHitRequired, total=False):
    """One search result.

    ``rank`` is the fused reciprocal-rank score: the field to sort, threshold or
    display when relevance matters. It is **not** comparable across queries —
    RRF sums ``1/(k + rank)`` per leg, so it is only meaningful within one
    response.
    """

    source: str | None
    type: str | None
    kind: str
    #: Writer-declared confidence; absent/null for ingested content with no score.
    confidence: float | None
    #: Fused RRF score. Compare within a response, never across responses.
    rank: float
    #: 1 when the body lives in blob storage and ``body`` is truncated or absent.
    is_large: int
    workspace_id: str
    #: The stored body. Present for small documents; absent when ``is_large``.
    body: str | None


class _SearchResponseRequired(TypedDict):
    results: list[SearchHit]


class SearchResponse(_SearchResponseRequired, total=False):
    #: The query the server actually ran, echoed back.
    query: str


class _PacketItemRequired(TypedDict):
    memory_id: str
    kind: str
    claim: str
    confidence: float
    authority: str
    applicability: Applicability


class PacketItem(_PacketItemRequired, total=False):
    """One retrieved memory inside a context packet, with provenance.

    ``condition`` is present **only** when ``applicability`` is
    ``"conditional"`` — the server omits the key entirely otherwise, so it is
    not merely nullable. ``source_ref`` is likewise omitted when a memory has no
    source.
    """

    condition: str
    source_ref: str


class EvidenceRef(TypedDict):
    memory_id: str
    source_uri: str | None
    source_span: str | None


class ConflictNote(TypedDict):
    memory_id_a: str
    memory_id_b: str
    note: str


class _ContextPacketRequired(TypedDict):
    summary: str
    facts: list[PacketItem]
    episodes: list[PacketItem]
    evidence: list[EvidenceRef]
    #: Warnings are instructions the caller should honour, not decoration.
    warnings: list[str]
    conflicts: list[ConflictNote]
    retrieved_ids: list[str]


class ContextPacket(_ContextPacketRequired, total=False):
    """Intent-aware context bundle returned by :meth:`Kiomon.retrieve_memory`."""

    procedure: PacketItem


class _LearningRequired(TypedDict):
    kind: LearningKind
    title: str
    body: str


class Learning(_LearningRequired, total=False):
    """A single learning to save through the write gate."""

    #: Short subheading shown on the memory card.
    excerpt: str
    #: 0–1. At or above 0.75 a write may activate immediately, unless the
    #: workspace is in review mode.
    confidence: float
    tags: list[str]
    #: Ids of memories this learning was derived from.
    source_refs: list[str]
    #: Supplying entities, claims and domain skips a paid extraction pass.
    entities: list[str]
    claims: list[str]
    domain: str


class BatchDraftMemory(TypedDict):
    id: str
    title: str
    kind: str


class _BatchDraftRequired(TypedDict):
    drafted: int
    #: Ids of memories the write gate activated immediately.
    promoted: list[str]
    #: Ids still awaiting review in the dashboard Review Inbox.
    pending: list[str]
    memories: list[BatchDraftMemory]


class BatchDraftResult(_BatchDraftRequired, total=False):
    #: Titles of learnings the write gate dropped rather than drafted.
    discarded: list[str]
    #: How many learnings were folded into an existing memory instead of inserted.
    merged: int
    #: Dashboard path to the Review Inbox. Sent only when ``pending`` is non-empty.
    review_inbox_url: str


class _GraphRequired(TypedDict):
    nodes: list[JsonObject]
    edges: list[JsonObject]


class GraphResult(_GraphRequired, total=False):
    """A page of the memory graph.

    ``hasMore`` is the pagination signal: without it a truncated walk is
    indistinguishable from a complete one.
    """

    #: Collapsed ``memory -> entity -> memory`` links — the structure retrieval
    #: walks. camelCase because that is the wire name.
    entityLinks: list[JsonObject]
    #: Nodes reachable from the centre before ``depth`` was applied.
    totalNodes: int
    hasMore: bool
    depth: int


class DocumentCounts(TypedDict):
    """Per-kind totals for the current filter.

    The API computes these on the first page only, to save a query on every
    subsequent page: with ``offset > 0`` every field comes back ``0``, not the
    real totals. ``total`` on the enclosing list is always correct.
    """

    total: int
    reference: int
    episodic: int
    semantic: int
    procedural: int
    pinned: int


class _DocumentListItemRequired(TypedDict):
    id: str
    source_url: str
    title: str | None
    created_at: str
    workspace_id: str
    kind: MemoryKind
    status: str
    source: str
    pinned: int


class DocumentListItem(_DocumentListItemRequired, total=False):
    """One row of the metadata-only list projection.

    ``body`` is deliberately absent — every list surface is metadata-only, and
    the body lives behind :meth:`Kiomon.get_memory`.
    """

    user_id: str
    author: str | None
    excerpt: str | None
    tags: str | None
    lang: str | None
    user_notes: str | None
    is_large: int
    updated_at: str | None
    pending_suggestion_count: int
    type: str | None
    thumbnail_url: str | None
    connector_metadata: str | None
    last_read_at: str | None
    confidence: float | None
    strength: float
    memory_metadata: str | None
    index_status: str
    index_error: str | None
    index_attempts: int
    index_changed_at: str | None
    ingest_status: str
    ingest_error: str | None
    ingest_pages: int | None
    ingest_changed_at: str | None
    enrich_status: str
    failed_at: str | None
    connector_id: str | None


class DocumentList(TypedDict):
    """One page of documents, with the counts for the current filter."""

    documents: list[DocumentListItem]
    limit: int
    offset: int
    total: int
    counts: DocumentCounts


class IngestResult(TypedDict, total=False):
    """The outcome of :meth:`Kiomon.ingest_document`.

    A save the API already had answers ``200`` with ``duplicate: true`` and the
    **existing** ``id`` instead of ``201``. Dedup is by source URL **within a
    workspace**, so re-ingesting the same page is a no-op rather than an error;
    a save with no ``source_url`` is given a synthetic unique one and never
    dedups. Check ``duplicate`` before treating the id as new.
    """

    id: str
    #: Tags the server resolved, including any it added itself. Absent on a dedup hit.
    tags: list[str]
    #: Present only on a dedup hit, where it is always ``true``.
    duplicate: bool
    #: Present only on a dedup hit: when the original was saved.
    created_at: str


class DeleteResult(TypedDict):
    """The outcome of :meth:`Kiomon.delete_document`."""

    deleted: bool


#: The state a listing selects. One type rather than two that must keep agreeing,
#: since documents and memories share the vocabulary.
MemoryStatus = DocumentStatus


class MemorySummary(TypedDict):
    """One row of the memory listing — the card shape, not the full document.

    The body is absent by design (read it with :meth:`Kiomon.get_memory`), and
    ``tags`` arrives as a JSON string, as it does everywhere else on the wire.
    """

    id: str
    title: str | None
    excerpt: str | None
    kind: MemoryKind
    status: str
    pinned: int
    tags: str | None
    source: str
    source_url: str
    workspace_id: str
    created_at: str
    confidence: float | None
    memory_metadata: str | None


class MemoryList(TypedDict):
    """One page of memories. ``status`` is echoed back, since it defaulted."""

    memories: list[MemorySummary]
    status: MemoryStatus
    limit: int
    offset: int
    total: int


class OutcomeItem(TypedDict):
    memory_id: str
    assessment: Assessment


#: Orientation context for a workspace. Shape is server-defined and open.
WorkspaceContext = dict[str, Any]


# ── Per-call options ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CallOptions:
    """Per-call overrides, mainly for retry and idempotency control."""

    #: Supply your own key to make a write retryable across processes.
    #: Generated when omitted.
    idempotency_key: str | None = None
    #: Override the client's ``max_retries`` for this call.
    max_retries: int | None = None
    #: Forwarded as ``X-Request-Id``, making a support request traceable.
    request_id: str | None = None
