# Releasing the Python SDK

One distribution is published from this directory:

| Package | PyPI name | Role |
|---|---|---|
| `kiomon-py/` | `kiomon` | the Python implementation |

The package is **not published yet**. Its PyPI name, `kiomon`, is the same name the
TypeScript SDK's unscoped npm alias claims; re-checked 2026-10-09, the PyPI name is still
unclaimed, as are `@kiomon/kiomon` and the unscoped npm `kiomon`.

## Preconditions

- The PyPI name `kiomon` is still available (verify before the first upload).
- A PyPI account with 2FA, and either a scoped API token or — preferred — a
  **Trusted Publisher** bound to the repository and a `pypi` environment so no token is
  stored in the repo.
- The version bumped in `src/kiomon/_version.py` **and** refreshed in `uv.lock`
  (`uv lock`). Publishing is permanent: a version can be yanked but never reused.
- `uv run pytest` green.

## Build and check

```bash
uv sync
uv run pytest
uv build                    # sdist + wheel into dist/
uv run --with twine twine check dist/*
```

## Publish

With a Trusted Publisher configured (GitHub Actions, `pypi` environment, `id-token:
write`), the upload is a `uv publish` step in CI. For a manual first release:

```bash
uv publish                  # uses UV_PUBLISH_TOKEN / twine-compatible env vars
```

or, equivalently:

```bash
uv run --with twine twine upload dist/*
```

## After publishing

- Add the PyPI badge/link to `README.md` and drop the "not published yet" note in
  `## Status`.
- Keep the version in step with the TypeScript SDK when the wire contract changes; the
  two surfaces are intended to be diffable against each other and against the MCP tools.

## Automation

`.github/workflows/ci.yml` runs on every push and pull request: tests on Python 3.10–3.13,
`ruff check` plus `mypy` on the lint job, and a build job that produces the sdist and wheel and
runs `twine check` on the metadata. There is no release workflow yet — the first publish should
use a Trusted Publisher bound to this repository and a `pypi` environment, so no token is ever
stored.
