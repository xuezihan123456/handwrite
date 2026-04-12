# Classroom Note Realism Design

Date: 2026-04-11
Status: approved for autonomous execution

## Goal

Reposition HandWrite from a generic handwriting demo into a classroom-note generator. The first visible upgrade is realism: the default output should feel less like a printed font pasted on paper and more like a natural, flowing student note.

## Product Decisions Locked In

- Primary audience: end users, not researchers or SDK consumers.
- First wedge: continuous paragraph classroom notes.
- Default style persona: natural and flowing, readable, slightly imperfect.
- Realism is more important than neatness.
- The first realism problem to fix is single-character print feel.
- Do not silently degrade. If realism is weak for some characters, tell the user.
- Do not auto-rewrite user text. Suggestions are advisory only.
- Keep multiple styles, but prioritize the default style first.
- Design the system for 2000+ character coverage later, but do not block this iteration on shipping the full asset body now.

## Scope For This Iteration

### In scope

1. Prototype-library foundation
   - Add a package-level prototype library concept for note-oriented handwritten glyph assets.
   - Ship a tiny distributable starter pack so the repo works out of the box.
   - Add a local build path so larger prototype packs can be created from local processed handwriting data later.

2. Better fallback path
   - When true generator weights are unavailable, prefer prototype glyphs first.
   - When a prototype glyph is unavailable, use a more handwritten-looking fallback path instead of a purely printed-looking render.
   - Keep the path compatible with future real model weights.

3. Realism inspection
   - Add a text inspection API that classifies characters before generation.
   - Mark which characters are covered by prototypes, supported by the model path, or likely to fall back to lower-realism rendering.
   - Generate human-readable advisory suggestions without changing user input.

4. Demo product loop
   - Add a classroom-note oriented precheck area in the Gradio demo.
   - Keep the multi-page preview/export flow.
   - Make the default style and demo copy reflect the classroom-note product positioning. Use an explicit default style constant so the built-in style order stays stable while the classroom-note default shifts to the flowing style.

5. Documentation and test coverage
   - Document the new note-generation workflow.
   - Add tests for prototype loading, inspection, and demo/report behavior.

### Out of scope

- Full 2000+ built-in prototype asset body.
- Personal handwriting upload/productization.
- Structured notes, Cornell notes, image mixing, formula layout.
- Automatic text rewriting.
- Claiming research-grade handwriting quality.

## Architecture

### 1. Prototype library

Add a new runtime module that loads handwritten glyph metadata from package assets or a user-specified directory.

Responsibilities:
- load manifest metadata
- resolve character -> glyph image path
- expose coverage stats for a text string
- support a starter built-in pack and future locally-built packs

Planned files:
- `src/handwrite/prototypes.py`
- `src/handwrite/assets/prototypes/default_note/manifest.json`
- `src/handwrite/assets/prototypes/default_note/*.png` (small starter pack)
- `scripts/build_prototype_library.py`

### 2. Style engine realism routing

Upgrade `StyleEngine` so character generation follows this order:

1. real generator output when weights are available
2. prototype glyph from the active prototype library
3. improved handwritten fallback rendering

The fallback renderer should still work without external assets. It should prefer more handwritten-looking system fonts when available, then apply controlled geometric and ink variation so uncovered characters still look less mechanical.

### 3. Inspection API

Add a new public inspection function, exposed from `src/handwrite/__init__.py`, that returns a structured report for note text.

Report fields should include:
- total characters
- unique non-whitespace characters
- prototype-covered characters
- model-covered characters
- lower-realism fallback characters
- advisory suggestions
- human-readable summary text / markdown

### 4. Demo loop

The demo should gain a note-quality precheck panel. Users can inspect first, then generate.

Expected user flow:
1. paste note text
2. click inspect/precheck
3. see realism report and suggestions
4. choose to generate anyway or edit input
5. export PNG/PDF as before

## Data and packaging constraints

- No large checkpoints or datasets are committed.
- Starter assets must stay small and distributable.
- Package data configuration must include the manifest and starter glyph assets.
- The build script for larger local packs should read local processed handwriting data and emit the same manifest format.

## Acceptance Criteria

1. `handwrite.inspect_text(...)` exists and returns a structured realism report.
2. `StyleEngine.generate_char(...)` can prefer prototype glyphs when available.
3. The default no-weight path produces visibly less mechanical output than the old plain fallback.
4. The demo exposes a note precheck/report area and keeps multi-page preview/downloads working.
5. The repo ships a starter prototype pack plus a script to build larger local packs.
6. Tests cover the new routing and report behavior.
7. README files describe HandWrite as a classroom-note generator instead of only a generic handwriting demo.

## Risks

- Small starter assets may not impress if users test outside the covered set immediately.
  - Mitigation: make the fallback path better and expose coverage before generation.
- Local prototype-library builder may be underspecified.
  - Mitigation: keep the first version narrowly scoped to processed dataset metadata.
- Demo complexity can sprawl.
  - Mitigation: use simple markdown/text reporting first, not complex in-input highlighting.
