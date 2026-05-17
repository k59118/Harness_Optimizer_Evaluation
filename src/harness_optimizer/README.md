## Backbone Agent Guide

`src/harness_optimizer` contains runtime adapters used by SHOR.
Adapters execute a `BackboneTask` and normalize trajectory output; SHOR owns
schema validation and result handling.

Included adapters:

- `openhands/`
- `claude_code_cli/`
- `codex_cli/`

Core files:

- `base.py`: shared types.
- `registry.py`: optimizer name registration and adapter loading.
- `io.py`, `prompting.py`, `errors.py`: helper utilities.
- `registry.py`: hardcoded adapter defaults and adapter registration.

To add a new adapter, create a package under `src/harness_optimizer/<name>/`,
then register its exact name, default settings, and a loader branch in
`registry.py`.
