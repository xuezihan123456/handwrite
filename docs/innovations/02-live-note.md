# Innovation 02 — Live Classroom-Note Animation

## Goal

Turn a paragraph or full note page into a "live writing" video where text
appears character-by-character on a styled page, exactly as a teacher would
write it on the blackboard. The deliverable is a list of grayscale frames
that can be exported as GIF or MP4 through the existing animation pipeline.

## Public Surface

- `LiveNoteEngine` — orchestrates layout + composition + cursor overlay.
- `NoteAnimationConfig` — dataclass holding style, paper, layout, font size, fps, wpm, pacing strategy, page size, margins, prototype pack, and cursor options.
- `live_note_video(text, output_path, ...) -> dict` — convenience function that renders + exports in one call and returns `{frame_count, duration_s, output_path}`.

## Architecture

`note_animator.py` does the heavy lifting in four stages:

1. **Layout** — text is wrapped into rows/columns that respect the chosen paper guide lines.
2. **Glyph caching** — each unique character is generated once with the existing `StyleEngine` and cached for reuse.
3. **Frame composition** — for each glyph we cross-fade the canvas from the previous state to the new state over the budget produced by `pacing.py`, giving the impression of progressive writing.
4. **Cursor overlay** — when enabled, `cursor.py` draws a pen-tip indicator (dark nib + soft halo) on every frame at the active glyph anchor.

`pacing.py` exposes three strategies (`linear`, `punctuation_pause`, `breath_pause`). Heavy punctuation slows the pen the most, light commas a little, and breath-style strategies add micro-pauses on every character.

## Constraints Honoured

- No edits to existing modules, configs, or dependencies.
- All file IO uses `pathlib.Path`; encoding is left to PIL/cv2 defaults.
- Tests stay under a 128x128 canvas and short text to keep runtime small.
