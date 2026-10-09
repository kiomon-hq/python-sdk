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
    "EvidenceRef",
    "GraphRelation",
    "GraphResult",
    "JsonObject",
    "Learning",
    "LearningKind",
    "ManageAction",
    "MemoryDocument",
    "MemoryKind",
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
    org_id: str | None


class WorkspaceList(TypedDict):
    workspaces: list[Workspace]


class MemoryDocument(TypedDict):
    """A stored memory/document. Mirrors the worker's ``Document`` contract.

    ``tags`` and ``headings`` arrive as JSON strings, not arrays.
    """

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
    source: str | None
    type: str | None
    kind: str
    #: Writer-declared confidence; absent/null for ingested content with no score.
    confidence: float | None


class _SearchResponseRequired(TypedDict):
    results: list[SearchHit]


class SearchResponse(_SearchResponseRequired, total=False):
    #: Present on some responses; kept optional rather than asserted.
    total: int


class PacketItem(TypedDict):
    """One retrieved memory inside a context packet, with provenance."""

    memory_id: str
    kind: str
    claim: str
    confidence: float
    authority: str
    applicability: Applicability
    condition: str | None
    source_ref: str | None


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
    review_inbox_url: str


class _GraphRequired(TypedDict):
    nodes: list[JsonObject]
    edges: list[JsonObject]


class GraphResult(_GraphRequired, total=False):
    #: Collapsed ``memory -> entity -> memory`` links — the structure retrieval
    #: walks. camelCase because that is the wire name.
    entityLinks: list[JsonObject]


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
