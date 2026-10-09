# kiomon

[![CI](https://github.com/kiomon-hq/python-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/kiomon-hq/python-sdk/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

The Kiomon Python SDK — persistent memory for AI agents and apps. Capture what your
users read and decide, retrieve it with provenance, and govern its lifecycle, from
your own product.

Zero runtime dependencies (the standard library only), typed end to end, Python 3.10+.

```bash
pip install kiomon
```

## Quickstart

```python
import os
from kiomon import Kiomon

kiomon = Kiomon(api_key=os.environ["KIOMON_API_KEY"])

# Retrieval first: ask for a context packet, not a list of documents.
packet = kiomon.retrieve_memory(
    intent="execute",
    query="how does our deploy pipeline work?",
)

print(packet["summary"])
for fact in packet["facts"]:
    print(f"- {fact['claim']} (confidence {fact['confidence']:.2f})")
# packet["procedure"] is the one best routine; packet["episodes"] the moments.
# packet["conflicts"] and packet["warnings"] tell you where the brain disagrees
# with itself — surface those instead of silently picking one.

# Tell the brain whether it helped. This is what makes ranking and decay honest.
kiomon.record_outcome(
    outcome="success",
    items=[
        {"memory_id": memory_id, "assessment": "helpful"}
        for memory_id in packet["retrieved_ids"]
    ],
)
```

## Async

`AsyncKiomon` is a native async client — no threads, no third-party HTTP library. It
speaks HTTP/1.1 over `asyncio` streams, with TLS from the standard library, so awaiting
a call never blocks the event loop and cancelling the await cancels the socket work
and closes the connection.

```python
import asyncio
from kiomon import AsyncKiomon

async def main():
    async with AsyncKiomon(api_key="sk_kiomon_…") as kiomon:
        packet = await kiomon.retrieve_memory(intent="plan", query="deployment")
        await kiomon.record_outcome(
            outcome="success",
            items=[{"memory_id": mid, "assessment": "helpful"} for mid in packet["retrieved_ids"]],
        )

asyncio.run(main())
```

Both clients share one request-planning core, so validation, retries, backoff,
idempotency and error codes are identical — only the I/O differs. The async client also
accepts an `async` token provider:

```python
kiomon = AsyncKiomon(token=refresh_kiomon_token)   # async def refresh_kiomon_token() -> str
```

Cancellation is real: `task.cancel()` aborts the request rather than leaving it running
behind the scenes. Timeouts are per attempt, and the default `AsyncioTransport` opens
one connection per request (`Connection: close`), so there is no pool to size or drain.

## Authentication

| Mode | Option | Header | Use for |
|---|---|---|---|
| Server-side key | `api_key` | `X-API-Key` | Your backend, worker, or agent service |
| Delegated OAuth | `token` | `Authorization: Bearer` | Acting on behalf of *your* user, via Kiomon's OAuth server |

```python
# A callable is resolved once per request, so token refresh needs no client rebuild.
kiomon = Kiomon(token=lambda: get_kiomon_token_for(user))
```

> **Delegated tokens are confined.** A `koa_…` connector token is restricted by the
> worker's default-deny allowlist to the memory routes. `request()` is not a general
> escape hatch in that mode.

## The 12 verbs

The Python method names are the hosted MCP tool names, so a capability you have in an
agent you also have here.

| Method | What it does |
|---|---|
| `retrieve_memory(intent=…, query=…, …)` | Intent-aware context packet with provenance, conflicts and warnings |
| `search_memory(query=…, kind=…, limit=…)` | Hybrid keyword + semantic search |
| `get_memory(id)` | One memory by id |
| `get_workspace_context(topic=…)` | Orientation pack — "where was I" |
| `get_latest_briefing()` | The most recent proactive briefing |
| `reflect_session(learnings=…)` | Save up to 25 learnings through the autonomous write gate |
| `draft_memory(learning)` | Save one learning |
| `manage_memory(id=…, action=…, value=…)` | `pin` · `unpin` · `archive` · `restore` · `forget` · `rate` · `approve` · `reject` · `revert` |
| `explore_memory_graph(id=…, relation=…, depth=…)` | Multi-hop traversal of the memory graph |
| `record_outcome(outcome=…, items=…)` | Report whether retrieved memories helped |
| `list_workspaces()` | Every workspace this credential can reach |
| `select_workspace(id)` | Bind the client to a workspace (chainable) |

Writes go through the same **write gate** as the dashboard: a high-confidence learning
may activate immediately, anything uncertain lands in the Review Inbox rather than
silently becoming knowledge.

## Workspaces

Pass `workspace_id` to the constructor, or rely on resolution: exactly one workspace
resolves silently, and more than one is an error rather than a guess — writing a memory
into the wrong brain is worse than failing.

```python
kiomon = Kiomon(api_key=key).select_workspace("ws_…")
```

## Errors

```python
from kiomon import KiomonError

try:
    kiomon.retrieve_memory(intent="plan", query="q")
except KiomonError as error:
    if error.code == "quota_exceeded":
        show_upgrade(error.needs_upgrade)          # HTTP 402 on Free
    elif error.retryable:
        retry_later()                              # 429, 5xx, timeouts, network
    elif error.code == "not_found":
        handle_missing()                           # may mean "no access" — see below
    else:
        raise
```

**Failure contract.** An invalid *argument* raises synchronously, before any network
call, so a programming error is never buried behind I/O. Anything that fails for
*remote* reasons raises `KiomonError` carrying `status`, `code`, `body`,
`needs_upgrade`, `request_id`, `retry_after` and `retryable`.

Codes, grouped by what you would actually do about them:

| Group | Codes |
|---|---|
| Bad request | `bad_request`, `validation_failed`, `payload_too_large` |
| Credentials | `unauthorized`, `account_pending_deletion`, `forbidden`, `insufficient_role`, `connector_scope_denied` |
| Absence | `not_found` — deliberately undifferentiated, see below |
| State | `conflict`, `duplicate`, `idempotency_conflict`, `idempotency_in_progress` |
| Limits | `quota_exceeded`, `pro_required`, `rate_limited` |
| Infrastructure | `service_unavailable`, `internal_error` |
| Raised by this SDK | `invalid_request` (bad arguments), `timeout`, `network_error` |

A code supplied by the server wins over the status mapping, and an unrecognised one
passes through unchanged, so a new server code never requires an SDK upgrade.

Two deliberate rough edges: **`not_found` does not distinguish "no access" from "does
not exist"**, because the API fails closed with 404 rather than confirming that a
resource exists — so treat it as "not available to you". And **`needs_upgrade` is
present but `False` on a Pro rate limit**, so the field is always safe to read when the
code is a limit.

## Retries and idempotency

Reads retry automatically on `429`, `5xx`, timeouts and network failure, with
exponential backoff and jitter. A `retry_after` value in the error body (or a
`Retry-After` header) is respected when present.

Every write carries an `Idempotency-Key` — generated per call, or passed in — and the
same key is reused across retries, so a replayed write cannot double-apply:

```python
from kiomon import CallOptions

kiomon.reflect_session(
    learnings=learnings,
    options=CallOptions(idempotency_key=f"session-{session_id}"),
)
```

Pass your own key when the retry may originate from a *different process*.

## Escape hatch

```python
stats = kiomon.request("/api/user/stats")
```

Every REST endpoint is reachable here, so the SDK never blocks you on a new route. Pass
`write=True` for a route that mutates state, so it gets an `Idempotency-Key` and is
retried safely:

```python
kiomon.request("/api/documents", method="POST", body={"…": "…"}, write=True)
```

## Custom transports

Each client talks to the network through a small seam. Synchronous `Kiomon` takes any
object with `send(request) -> Response`; `AsyncKiomon` takes any object with
`async def send(request) -> Response`. The defaults are the standard library:
`UrllibTransport` (sync, honours proxy environment variables) and `AsyncioTransport`
(async, one connection per request).

```python
from kiomon import AsyncKiomon, Kiomon, Request, Response

class MyTransport:
    def send(self, request: Request) -> Response:        # sync
        ...  # return Response(status, headers, body)

class MyAsyncTransport:
    async def send(self, request: Request) -> Response:  # async
        ...

kiomon = Kiomon(api_key=key, transport=MyTransport())
kiomon = AsyncKiomon(api_key=key, transport=MyAsyncTransport())
```

Both default transports accept `ssl_context=…` when you need a pinned CA bundle, and
both clients are context managers (`with` / `async with`) that call the transport's
`close()` / `aclose()` when it has one.

## Parity with the TypeScript SDK and MCP

- Field names are the **wire names** (`workspace_id`, `memory_id`, `source_refs`), so an
  SDK call can be diffed against the equivalent MCP tool call. The one exception is
  `GraphResult["entityLinks"]`, which is camelCase on the wire.
- Method names are the **MCP tool names** (`retrieve_memory`, `manage_memory`, …).
- Local validation matches the TypeScript SDK exactly: constructor credentials, ids and
  queries, the 25-learning ceiling, the 0–1 confidence range, the manage-action
  vocabulary, and the retrieval limit ceiling. Everything else is the server's contract
  and is validated there.
- Python-specific conventions: `None` means "not set" and the key is omitted from the
  wire (matching the TypeScript SDK's `undefined`); timeouts are **seconds**, not
  milliseconds, and the parameter is `timeout=` rather than `timeoutMs=`.
- The async client is native asyncio rather than thread offloading: cancellation is
  real, and an invalid argument surfaces when the call is awaited (nothing is sent)
  rather than before the call as it does in the sync client.

## Development

```bash
uv sync                       # create the venv and install the dev group
uv run pytest                 # the test suite
uv run ruff check .           # lint
uv run mypy                   # types
```

No runtime dependencies, so the venv holds only the tooling. `mypy` runs in strict mode
over `src/kiomon`; the tests use `pythonpath = ["src"]` from `pyproject.toml`, so no
install step is needed before running them.

## Status

`0.1.0`. Complete against the shipped API. The sync and async clients share one
request-planning core and are verified with `uv run pytest` (unit, plus
loopback-socket integration tests for both the urllib and asyncio transports,
including chunked and EOF-framed responses, redirects, timeouts and cancellation).
The package is **not published yet**; until it is, install it from a checkout with
`uv sync` / `pip install .`.

MIT © Kiomon
