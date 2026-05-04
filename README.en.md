# HandWrite English Guide

## Project Overview

HandWrite is an **AI-powered Chinese handwriting generator** with **15 innovation modules** spanning from stroke dynamics to AR augmentation. It works as both a classroom-note generation tool and an extensible handwriting synthesis research platform.

The repository includes data preprocessing, training scaffolding, inference wrappers, page composition, PNG/PDF export, a Gradio demo, and note-realism precheck. It also integrates 15 innovation modules covering stroke dynamics, animation, semantic layout, formula rendering, style mixing, collaborative writing, quality assurance, handwriting recognition, grading, temporal evolution, and AR enhancement.

> No real pretrained weights are bundled. When valid weights are missing, runtime falls back to prototype-backed rendering to keep the product loop usable.

## Environment and Installation

- Python 3.9+
- PyTorch, Pillow, OpenCV, Gradio

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"
pytest
python demo/app.py
```

## Core API

### List built-in styles

```python
import handwrite
print(handwrite.list_styles())
```

### Inspect note text before generation

```python
import handwrite

report = handwrite.inspect_text(
    "Today we reviewed Newton's second law and worked through two examples.",
    style="行书流畅",
)
print(report["summary"])
for advisory in report["suggestions"]:
    print(advisory)
```

### Generate one note page

```python
import handwrite

page = handwrite.generate(
    "Today we reviewed Newton's second law and worked through two examples.",
    style="行书流畅", paper="横线纸", layout="自然", font_size=80,
)
page.save("note_page.png")
```

### Generate multiple note pages

```python
import handwrite

pages = handwrite.generate_pages(
    "Classroom notes content " * 120,
    style="行书流畅", paper="横线纸", layout="自然", font_size=80,
)
for index, page in enumerate(pages, start=1):
    page.save(f"note_{index:03d}.png")
```

### Export PNG / PDF

```python
import handwrite

handwrite.export(page, "output/page.png", format="png", dpi=300)
handwrite.export(page, "output/page.pdf", format="pdf", dpi=300)
handwrite.export_pages(pages, "output/note.pdf", format="pdf", dpi=300)
```

### Use a local prototype pack

```python
import handwrite

prototype_pack = "data/prototypes/my_note_pack"

report = handwrite.inspect_text(
    "Today we reviewed Newton's second law.",
    style="行书流畅", prototype_pack=prototype_pack,
)
page = handwrite.generate(
    "Today we reviewed Newton's second law.",
    style="行书流畅", paper="横线纸", layout="自然", font_size=80,
    prototype_pack=prototype_pack,
)
```

## 15 Innovation Modules

### 1. Personalization

Extract handwriting features from sample images (stroke width, slant angle, connectivity, ink distribution), generate style vectors, and synthesize personalized glyph packs.

```python
from handwrite.personalization import SampleAnalyzer, StyleExtractor, GlyphSynthesizer

analyzer = SampleAnalyzer()
features = analyzer.analyze("sample.png")

extractor = StyleExtractor()
style = extractor.extract(features)

synth = GlyphSynthesizer(style)
synth.synthesize_pack(output_dir, charset="Hello")
```

### 2. Dynamics

Simulate pen pressure, ink flow, and writing speed during the writing process for realistic stroke dynamics.

```python
from handwrite.dynamics import DynamicsEngine

engine = DynamicsEngine()
result = engine.simulate(stroke_data)
# Returns pressure_map, ink_flow, speed_profile
```

### 3. Animation

Stroke-order detection with Bezier trajectory generation. Supports frame-by-frame animation rendering with GIF and MP4 export.

```python
from handwrite.animation import StrokeOrderDetector, TrajectoryGenerator, AnimationEngine

detector = StrokeOrderDetector()
order = detector.detect(char_image)

generator = TrajectoryGenerator()
trajectory = generator.generate(order)

engine = AnimationEngine()
frames = engine.render(trajectory)
engine.export_gif(frames, "writing.gif")
```

### 4. Semantic Layout

Analyze text semantic structure, intelligently plan page layout, and render semantic annotations (headings, paragraphs, lists).

```python
from handwrite.semantic import TextAnalyzer, LayoutPlanner, SemanticComposer

analyzer = TextAnalyzer()
structure = analyzer.analyze("Classroom notes content...")

planner = LayoutPlanner()
layout = planner.plan(structure, page_size=(2480, 3508))

composer = SemanticComposer()
page = composer.compose(layout)
```

### 5. Formula Rendering

Parse and render LaTeX math formulas and chemical equations, including matrices, fractions, subscripts, and superscripts.

```python
from handwrite.formula import LatexParser, ChemistryParser, FormulaRenderer

parser = LatexParser()
formula = parser.parse(r"\frac{-b \pm \sqrt{b^2-4ac}}{2a}")

chem = ChemistryParser()
equation = chem.parse("2H2 + O2 -> 2H2O")

renderer = FormulaRenderer()
img = renderer.render(formula)
```

### 6. Style Mixing

Blend multiple handwriting styles, perform style transfer, and create unique personal styles through interpolation.

```python
from handwrite.style_mixing import StyleMixer, StyleTransfer

mixer = StyleMixer()
blended = mixer.mix(style_a, style_b, ratio=0.6)

transfer = StyleTransfer()
transferred = transfer.transfer(source_style, target_content)
```

### 7. Paper Templates

6 built-in paper templates (Cornell notes, English practice, error notebook, essay grid, staff paper, mind map) with custom paper support.

```python
from handwrite.papers import PaperRegistry, PaperRenderer

registry = PaperRegistry()
paper = registry.get("cornell")

renderer = PaperRenderer()
img = renderer.render(paper)
```

Built-in templates:
- `cornell` - Cornell notes (cue + note + summary regions)
- `english_practice` - English practice paper (four-line three-grid)
- `error_notebook` - Error correction notebook (question + wrong + correct + reflection)
- `essay_grid` - Chinese essay grid paper
- `staff` - Music staff paper
- `mind_map` - Mind map with central node and branches

### 8. OCR Style Extraction

Extract handwriting style from scanned images and generate prototype fonts for personalized generation.

```python
from handwrite.ocr_style import ImagePreprocessor, CharacterSegmenter, StyleExtractor

preprocessor = ImagePreprocessor()
clean = preprocessor.preprocess(scanned_image)

segmenter = CharacterSegmenter()
chars = segmenter.segment(clean)

extractor = StyleExtractor()
style = extractor.extract(chars)
```

### 9. Collaborative Writing

Multi-writer collaborative writing with automatic segment assignment and style blending across contributors.

```python
from handwrite.collaboration import CollaborativeComposer, StyleBlender

composer = CollaborativeComposer()
segments = composer.assign_segments(contributors, content)

blender = StyleBlender()
result = blender.blend(segments)
```

### 10. Quality Assurance

Evaluate generated handwriting quality across authenticity and naturalness dimensions with improvement suggestions.

```python
from handwrite.quality import QualityEngine

engine = QualityEngine()
report = engine.evaluate(generated_image)
# Returns authenticity_score, naturalness_score, suggestions
```

### 11. Text Summarization

Automatic key information extraction with mind-map and outline layout generation.

```python
from handwrite.summary import TextSummarizer, MindMapLayout

summarizer = TextSummarizer()
summary = summarizer.summarize(long_text)

layout = MindMapLayout()
mind_map = layout.generate(summary)
```

### 12. Digitization

OCR recognition of handwritten text with style-preserving round-trip editing — recognize, modify, and regenerate in the original style.

```python
from handwrite.digitization import HandwritingRecognizer, RoundTripEngine

recognizer = HandwritingRecognizer()
text = recognizer.recognize(handwritten_image)

engine = RoundTripEngine()
edited_image = engine.edit(handwritten_image, original_text=text, new_text="Corrected text")
```

### 13. Grading

Automatically detect errors in handwritten assignments, generate annotations, scores, and feedback.

```python
from handwrite.grading import GradingEngine

engine = GradingEngine()
result = engine.grade(submission_image, reference_text="Correct answer")
# Returns score, errors, annotations, feedback
```

### 14. Temporal Evolution

Simulate handwriting changes across different ages and skill levels, with historical style reproduction.

```python
from handwrite.temporal import TemporalEngine

engine = TemporalEngine()
aged = engine.simulate_age(glyph, target_age=10)
skilled = engine.simulate_skill(glyph, skill_level=0.8)
```

### 15. AR Enhancement

Paper detection, perspective transform, and texture blending for augmented reality handwriting overlay.

```python
from handwrite.ar import AREngine

engine = AREngine()
detected = engine.detect_paper(camera_frame)
transformed = engine.transform(detected)
blended = engine.blend(transformed, digital_content)
```

## CLI Scripts

```bash
# Data preparation
python scripts/download_data.py --scan_dir downloads --raw_dir data/raw

# Preprocessing
python scripts/preprocess.py --raw_dir data/raw/HWDB1.0trn_gnt --output_dir data/processed

# Build prototype library
python scripts/build_prototype_library.py --metadata data/processed/metadata.json --output_dir data/prototypes/default_note

# Classroom note session
python scripts/note_session.py --preset 牛顿定律复习 --output_dir output/session

# Training
python scripts/train.py --data_dir data/processed --styles_file data/processed/selected_styles.json --output_dir weights

# Evaluation
python scripts/evaluate.py --output-dir evaluation

# Demo
python demo/app.py
```

## Project Structure

```
src/handwrite/
├── animation/          # Stroke order, trajectory, frame rendering, GIF/MP4 export
├── ar/                 # Paper detection, perspective transform, texture blending
├── collaboration/      # Contributor management, segment assignment, style blending
├── data/               # CASIA parser, charsets, dataset utilities
├── digitization/       # OCR recognition, style preservation, round-trip editing
├── dynamics/           # Pressure, ink flow, speed simulation
├── engine/             # Generator, discriminator, training engine
├── formula/            # LaTeX/chemistry parsing, layout, rendering
├── grading/            # Error detection, annotation, scoring, feedback
├── ocr_style/          # Image preprocessing, character segmentation, style extraction
├── papers/             # Paper template registry, built-in papers, renderer
├── personalization/    # Sample analysis, style extraction, glyph synthesis
├── quality/            # Authenticity scoring, naturalness scoring, improvement advisor
├── semantic/           # Text analysis, layout planning, annotation rendering
├── style_mixing/       # Style vectors, mixing, transfer, interpolation
├── summary/            # Text summarization, mind map layout, outline layout
├── temporal/           # Age profiles, historical style, skill simulation
├── composer.py         # Page composition engine
├── exporter.py         # PNG/PDF export
├── prototypes.py       # Prototype pack management
└── styles.py           # Style registry
```

## Current Limitations

1. **No pretrained weights** - Uses prototype-backed fallback when weights are unavailable
2. **Limited starter pack** - Built-in prototype pack is small; larger coverage requires local building
3. **No automatic text rewriting** - Precheck reports and suggests, but never modifies user text
4. **Lightweight evaluation** - Includes basic metrics, not a full research-grade benchmark

## Suggested Workflow

1. `pip install -e ".[dev]"` - Install dependencies
2. `pytest` - Run tests (623+ tests passing)
3. `handwrite.inspect_text(...)` - Precheck text coverage
4. `python demo/app.py` - Try the full product loop
5. Prepare `.gnt` data and Chinese fonts
6. `scripts/build_prototype_library.py` - Build a larger prototype pack
7. Run training/evaluation scripts for higher quality
