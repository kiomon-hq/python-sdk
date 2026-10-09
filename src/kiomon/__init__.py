"""Kiomon Python SDK — persistent memory for AI agents and apps.

Capture what your users read and decide, retrieve it with provenance, and
govern its lifecycle, from your own product.

Example:
    >>> from kiomon import Kiomon
    >>>
    >>> kiomon = Kiomon(api_key="sk_kiomon_…")
    >>> packet = kiomon.retrieve_memory(intent="execute", query="how do we deploy?")
    >>> kiomon.record_outcome(
    ...     outcome="success",
    ...     items=[{"memory_id": mid, "assessment": "helpful"} for mid in packet["retrieved_ids"]],
    ... )
"""

from ._async import AsyncKiomon
from ._async_transport import AsyncioTransport, AsyncTransport
from ._client import DEFAULT_BASE_URL, DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, Kiomon, TokenProvider
from ._transport import Request, Response, Transport, UrllibTransport
from ._version import __version__
from .errors import (
    DEFAULT_CODE_BY_STATUS,
    KiomonError,
    KiomonErrorCode,
    code_for_status,
    error_from_response,
    is_kiomon_error,
)
from .types import (
    Applicability,
    Assessment,
    BatchDraftMemory,
    BatchDraftResult,
    CallOptions,
    ConflictNote,
    ContextPacket,
    EvidenceRef,
    GraphRelation,
    GraphResult,
    JsonObject,
    Learning,
    LearningKind,
    ManageAction,
    MemoryDocument,
    MemoryKind,
    Outcome,
    OutcomeItem,
    PacketItem,
    RetrievalIntent,
    RiskLevel,
    SearchHit,
    SearchResponse,
    Workspace,
    WorkspaceContext,
    WorkspaceList,
)

__all__ = [
    # Clients
    "AsyncKiomon",
    "Kiomon",
    # Async transport seam
    "AsyncioTransport",
    "AsyncTransport",
    # Errors
    "DEFAULT_CODE_BY_STATUS",
    "KiomonError",
    "KiomonErrorCode",
    "code_for_status",
    "error_from_response",
    "is_kiomon_error",
    # Transport seam
    "Request",
    "Response",
    "TokenProvider",
    "Transport",
    "UrllibTransport",
    # Constants
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT",
    "__version__",
    # Types
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
