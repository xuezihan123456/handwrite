# Innovation #10: 3D Pen-Tip Dynamics

## Overview

Innovation #10 lifts HandWrite's 2D dynamics pipeline into a richer 3D pen-tip state space. Every sample on a stroke is upgraded from a plain `(x, y)` coordinate to an eight-field `PenSample3D` capturing position, time, pressure, X / Y tilt, barrel rotation, and instantaneous velocity. Strokes are bundled into `PenStroke3D` containers that carry free-form metadata (sample-rate hints, author tags, tool identifiers).

The new `Pen3DSimulator` takes any 2D point sequence and produces a fully populated 3D stroke. Pressure is inversely related to pen speed so fast strokes leave a lighter mark; tilt evolves along smooth low-frequency curves with a small amount of paper-surface friction jitter; barrel rotation drifts monotonically through `[0, 2*pi)`. Everything is deterministic when you set `seed`.

## Interoperability

`will-lite` JSON schema. `export_will_json` serializes strokes to a plain dictionary; `import_will_json`, `save_will_file`, `load_will_file` complete the round-trip. No Wacom SDK required.

## Replay

`replay_to_image` walks a sequence of 3D strokes and renders them to a PIL image. Stroke width is modulated by a combination of pressure **and** tilt magnitude.

## Use Cases

* **Calligraphy / book training**: replay famous masters' strokes with the original tilt and pressure curves.
* **Stroke rehabilitation**: physiotherapists can record patient strokes, export them as WILL-lite JSON.
* **Dataset augmentation**: synthesise diverse pen-tilt variants from a single 2D template.
