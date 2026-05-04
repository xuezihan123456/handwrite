"""Extract handwriting style from a scanned image and generate a prototype pack.

Usage:
    python scripts/extract_style_from_scan.py --image scan.jpg
    python scripts/extract_style_from_scan.py --image scan.jpg --text "你好世界"
    python scripts/extract_style_from_scan.py --image scan.jpg --output_dir data/prototypes/my_style

Pipeline:
    preprocess -> segment -> extract style -> generate prototypes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from handwrite.ocr_style.image_preprocessor import ImagePreprocessor
from handwrite.ocr_style.character_segmenter import CharacterSegmenter
from handwrite.ocr_style.style_extractor import StyleExtractor
from handwrite.ocr_style.prototype_generator import PrototypeGenerator


def run(
    image_path: str | Path,
    *,
    text: str | None = None,
    output_dir: str | Path | None = None,
    pack_name: str | None = None,
    writer_id: str = "scan",
    glyph_size: int = 256,
) -> dict:
    """Run the full extraction pipeline.

    Parameters
    ----------
    image_path:
        Path to the scanned handwriting image.
    text:
        Optional ground-truth text.  When provided, characters are labelled
        with the actual text.  Otherwise positional labels are used.
    output_dir:
        Output directory for the prototype pack.  Defaults to
        ``data/prototypes/scan_{timestamp}/``.
    pack_name:
        Pack name for the manifest.  Defaults to ``scan_{timestamp}``.
    writer_id:
        Writer identifier stored in manifest entries.
    glyph_size:
        Output glyph image size (default 256).

    Returns
    -------
    dict
        Summary including style features and pack metadata.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "prototypes" / f"scan_{timestamp}"
    else:
        output_dir = Path(output_dir)
    if pack_name is None:
        pack_name = f"scan_{timestamp}"

    # Step 1: Preprocess
    print("[1/4] 预处理图像...")
    preprocessor = ImagePreprocessor()
    preprocess_result = preprocessor.preprocess(image_path)
    print(
        f"  原始尺寸: {preprocess_result.original_shape}, "
        f"倾斜角: {preprocess_result.skew_angle:.1f}°, "
        f"透视校正: {'是' if preprocess_result.perspective_corrected else '否'}"
    )

    # Step 2: Segment
    print("[2/4] 分割字符...")
    segmenter = CharacterSegmenter()
    chars = segmenter.segment(preprocess_result.image)
    print(f"  分割到 {len(chars)} 个字符区域")

    if not chars:
        print("  警告: 未检测到任何字符区域，请检查图像质量")
        return {
            "image_path": str(image_path),
            "num_chars": 0,
            "style_features": {},
            "pack_summary": None,
        }

    # Step 3: Extract style features
    print("[3/4] 提取风格特征...")
    extractor = StyleExtractor()
    features = extractor.extract(chars)
    print(f"  笔画宽度: {features.stroke_width_mean:.1f} +/- {features.stroke_width_std:.1f}")
    print(f"  宽高比: {features.aspect_ratio_mean:.2f}")
    print(f"  墨迹浓度: {features.ink_density_mean:.2f}")

    # Step 4: Generate prototypes
    print("[4/4] 生成原型库...")

    # Build labels list from optional text
    labels: list[str] | None = None
    if text is not None:
        # Filter out whitespace from the text to align with segments
        text_chars = [c for c in text if not c.isspace()]
        if len(text_chars) >= len(chars):
            labels = text_chars[: len(chars)]
        else:
            # Text is shorter than detected chars -- label what we can
            labels = text_chars + [f"_pos{i}" for i in range(len(text_chars), len(chars))]

    generator = PrototypeGenerator(
        glyph_size=glyph_size,
        writer_id=writer_id,
    )
    pack_summary = generator.generate(
        chars=chars,
        labels=labels,
        output_dir=output_dir,
        pack_name=pack_name,
    )

    print(f"  输出目录: {output_dir}")
    print(f"  生成 {pack_summary['glyph_count']} 个原型字形")
    print(f"  字符列表: {''.join(pack_summary['chars'][:30])}")
    if len(pack_summary['chars']) > 30:
        print(f"  ... 共 {len(pack_summary['chars'])} 个")

    return {
        "image_path": str(image_path),
        "num_chars": len(chars),
        "style_features": features.as_dict(),
        "pack_summary": pack_summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从扫描手写笔记中提取笔迹风格，生成原型字形库",
    )
    parser.add_argument(
        "--image", type=Path, required=True,
        help="扫描图像路径",
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="可选的对应文本（用于标注分割出的字符）",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="输出目录，默认 data/prototypes/scan_{timestamp}",
    )
    parser.add_argument(
        "--pack_name", type=str, default=None,
        help="原型包名称",
    )
    parser.add_argument(
        "--writer_id", type=str, default="scan",
        help="书写者 ID（默认 scan）",
    )
    parser.add_argument(
        "--glyph_size", type=int, default=256,
        help="输出字形尺寸（默认 256）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run(
        image_path=args.image,
        text=args.text,
        output_dir=args.output_dir,
        pack_name=args.pack_name,
        writer_id=args.writer_id,
        glyph_size=args.glyph_size,
    )
    print("\n=== 完成 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
