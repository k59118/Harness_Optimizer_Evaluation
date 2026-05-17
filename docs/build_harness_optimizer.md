# Harness Optimizer Implementation

This guide explains the minimum contract for adding your own harness optimizer
to SHOR evaluation.

## What You Implement

Implement one adapter under:

```text
src/harness_optimizer/<your_optimizer_name>/
```

Your adapter must expose a class that can run a SHOR task and return a
`BackboneRunResult`.

The shared types are defined in:

```text
src/harness_optimizer/base.py
```

## Input

SHOR passes your adapter a `BackboneTask`.

Important fields:

- `task.workspace_root`: working directory for your optimizer process
- `task.instructions`: the prompt/instructions to give your optimizer
- `task.artifact_paths`: output files your optimizer must create
- `task.trajectory_path`: optional debug/trajectory output path
- `task.limits`: runtime limits such as timeout

For SHOR evaluation, the required output path is:

```python
task.artifact_paths["shor_json"]
```

Your optimizer should read the task instructions, inspect the provided files,
and write its answer to that path.

## Output

Your adapter must return:

```python
BackboneRunResult(...)
```

Use:

- `status="ok"` when the optimizer finished and wrote the required output
- `status="error"` when the optimizer failed
- `status="timeout"` when the run timed out

At minimum, fill:

- `status`
- `message`
- `trajectory_path`

Fill these when available:

- `n_steps`
- `cost`
- `token_usage`
- `raw_stdout_path`
- `raw_stderr_path`

## Required Behavior

Your adapter is responsible for:

1. Preparing any runtime files it needs.
2. Running your optimizer in `task.workspace_root`.
3. Passing `task.instructions` to your optimizer.
4. Enforcing the timeout if possible.
5. Verifying that every file in `task.artifact_paths` exists before returning
   `status="ok"`.
6. Returning a useful error message when the run fails.

SHOR is responsible for:

1. Creating the task.
2. Staging input files.
3. Validating the final output.
4. Writing public result metadata.

## Register Your Optimizer

Register your adapter in:

```text
src/harness_optimizer/registry.py
```

Add:

1. The optimizer name in `BACKBONE_NAMES`.
2. A default config in `DEFAULT_BACKBONE_CONFIGS`.
3. A loader branch in `load_backbone`.

After registration, the optimizer name should work here:

```bash
python src/shor/run_shor.py --optimizer your_optimizer_name

# Optional: run only the first 10
python src/shor/run_shor.py --optimizer your_optimizer_name --limit 10
```

## Built-In Examples

Use these as references:

- `src/harness_optimizer/openhands/`
- `src/harness_optimizer/claude_code_cli/`
- `src/harness_optimizer/codex_cli/`

## Minimal Checklist

Before running SHOR evaluation, confirm:

- your optimizer is registered in `registry.py`
- `python src/shor/run_shor.py --optimizer your_optimizer_name --help` works
- your adapter writes `task.artifact_paths["shor_json"]`
- your adapter returns `status="ok"` only after the required output exists
