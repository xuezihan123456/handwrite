# Classroom Note Realism Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first classroom-note product loop: prototype-backed character routing, realism inspection, a demo precheck report, and docs/tests that support future large-scale note coverage.

**Architecture:** Add a prototype library layer between true model inference and fallback rendering. Expose inspection/reporting in the public API, wire it into the Gradio demo, and add a local builder script so the starter pack can scale into a larger private note library later.

**Tech Stack:** Python 3.9+, Pillow, pytest, setuptools package data, Gradio.

---

## File Map

### Core runtime
- Create: `src/handwrite/prototypes.py`
- Modify: `src/handwrite/engine/model.py`
- Modify: `src/handwrite/__init__.py`
- Modify: `src/handwrite/styles.py`
- Modify: `pyproject.toml`

### Assets and builder
- Create: `src/handwrite/assets/prototypes/default_note/manifest.json`
- Create: `src/handwrite/assets/prototypes/default_note/*.png`
- Create: `scripts/build_prototype_library.py`

### Demo
- Modify: `demo/app.py`

### Tests
- Create: `tests/test_prototypes.py`
- Modify: `tests/test_engine.py`
- Modify: `tests/test_package.py`
- Modify: `tests/test_demo.py`

### Docs
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `README.en.md`
- Create: `docs/superpowers/specs/2026-04-11-classroom-note-realism-design.md`

---

### Task 1: Add the prototype library foundation
- [ ] Write failing tests for manifest loading, starter-pack lookup, and coverage inspection.
- [ ] Implement `src/handwrite/prototypes.py` with manifest loading, glyph lookup, and text coverage summary.
- [ ] Add starter package data under `src/handwrite/assets/prototypes/default_note/`.
- [ ] Update packaging so the manifest and starter glyph assets ship with the package.
- [ ] Run focused tests for the new prototype module.

### Task 2: Upgrade the style engine routing
- [ ] Write failing tests that prove prototype glyphs are preferred before low-realism fallback.
- [ ] Extend `StyleEngine` to load an active prototype library and expose per-character routing metadata.
- [ ] Improve the no-weight fallback rendering path so uncovered characters look less mechanical.
- [ ] Keep the existing real-weight path intact.
- [ ] Run focused engine tests and adjust regressions.

### Task 3: Expose the inspection/report API
- [ ] Write failing public-API tests for `handwrite.inspect_text(...)`.
- [ ] Add structured report generation in `src/handwrite/__init__.py` using engine/prototype coverage information.
- [ ] Include advisory suggestions and a ready-to-render summary/report text.
- [ ] Make the default style a note-oriented flowing style without removing other styles or changing built-in style ordering.
- [ ] Run focused package/API tests.

### Task 4: Wire the precheck loop into the Gradio demo
- [ ] Write failing demo tests for the new precheck report outputs and default note workflow.
- [ ] Add an inspect/precheck function in `demo/app.py`.
- [ ] Show the report in the demo while preserving multi-page preview/downloads.
- [ ] Update default labels/copy toward the classroom-note product story.
- [ ] Run focused demo tests.

### Task 5: Add the local builder path
- [ ] Write failing tests for a minimal prototype-library build flow from local metadata/images.
- [ ] Implement `scripts/build_prototype_library.py` so a user can emit a manifest-compatible pack from local processed handwriting data.
- [ ] Keep the builder narrow and deterministic, using one handwriting sample per character.
- [ ] Run focused builder tests.

### Task 6: Update docs and run verification
- [ ] Rewrite README messaging around classroom-note generation, starter-pack realism, and precheck workflow.
- [ ] Document the starter-pack limitation and the local builder path for larger coverage.
- [ ] Run targeted pytest commands for all changed areas.
- [ ] Run a combined regression command over prototypes, engine, package, demo, and builder paths.
- [ ] Record results in `progress.md` and close the task plan phase.


### Task 7: Custom prototype-pack runtime loop
- [x] Add a public/runtime path to load a local prototype pack built by `scripts/build_prototype_library.py`.
- [x] Make engine caching/config selection aware of the active prototype pack so inspect/generate use the same source.
- [x] Expose the active prototype source in inspection reports.
- [x] Add tests for default-pack vs custom-pack behavior.

### Task 8: Demo custom-pack workflow
- [x] Add an optional demo input for a local prototype pack directory or manifest.
- [x] Pass the selected pack into precheck and generation.
- [x] Show which pack is active in the report so the user knows whether they are using the built-in starter pack or a local expanded pack.
- [x] Add focused demo tests for the new flow.

### Task 9: Docs + verification for custom packs
- [x] Update README files to document how a locally built prototype pack is used at runtime and in the demo.
- [x] Re-run focused tests and the full `tests/` suite.
- [x] Re-run editable install verification to ensure runtime package-data remains intact.
