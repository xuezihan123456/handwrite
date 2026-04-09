# Repository Guidelines

## Project Structure & Module Organization
This repository is currently docs-first. `handwrite-design.md` defines product scope and architecture; `handwrite-implementation-guide.md` defines the target code layout. As implementation is added, keep code under `src/handwrite/`, with `src/handwrite/data/` for dataset and preprocessing logic, `src/handwrite/engine/` for models and training, `demo/` for the Gradio app, `scripts/` for runnable utilities, and `tests/` plus `tests/fixtures/` for validation assets. Keep large checkpoints in `weights/`, but do not commit binaries.

## Build, Test, and Development Commands
The runnable project scaffold is planned but not fully present yet. Use these commands once the package skeleton exists:

```bash
python -m venv .venv
pip install -e ".[dev]"
pytest
python demo/app.py
python scripts/train.py
```

`python -m venv .venv` creates an isolated environment. `pip install -e ".[dev]"` installs the package with development extras. `pytest` runs the test suite. `python demo/app.py` starts the local demo, and `python scripts/train.py` launches model training.

## Coding Style & Naming Conventions
Target Python 3.9+ and follow PEP 8 with 4-space indentation. Use `snake_case` for modules, functions, and script names, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants such as character-set definitions. Keep the public API minimal in `src/handwrite/__init__.py`, prefer explicit type hints on public functions, and keep modules narrowly scoped. No formatter or linter is configured yet, so keep imports tidy and formatting consistent by review.

## Testing Guidelines
Use `pytest` and name files `test_*.py`, for example `tests/test_exporter.py`. Mirror the package structure where practical, and place reusable sample inputs under `tests/fixtures/`. New modules should include at least one unit test; changes affecting inference, composition, or export paths should also include a smoke-path check.

## Commit & Pull Request Guidelines
There is no Git history in this workspace yet, so establish the convention now: use short, imperative commit subjects such as `data: add CASIA parser` or `demo: wire Gradio preview`. Pull requests should reference the relevant section in `handwrite-design.md` or `handwrite-implementation-guide.md`, list the commands used for validation, and include screenshots or generated samples for demo or rendering changes.

## Security & Configuration Tips
Do not commit datasets, model weights, or machine-specific paths. Keep `.gnt`, `.pt`, `.pth`, and large generated evaluation images out of version control, and document any required local data directories in the PR description.
