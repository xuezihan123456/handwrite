"""CLI tool for personalized handwriting learning.

Usage:
    python scripts/personalize.py --sample sample.png --output_dir data/prototypes/my_style
    python scripts/personalize.py --sample s1.png --sample s2.png --output_dir data/prototypes/my_style
    python scripts/personalize.py --analyze sample.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="个性化手写学习：分析手写样本，提取风格，生成个性化原型字形包"
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # --- analyze ---
    p_analyze = sub.add_parser("analyze", help="分析手写样本图片，输出特征报告")
    p_analyze.add_argument(
        "samples",
        nargs="+",
        help="手写样本图片路径（支持多张）",
    )
    p_analyze.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出",
    )

    # --- extract ---
    p_extract = sub.add_parser("extract", help="从样本中提取风格向量")
    p_extract.add_argument(
        "samples",
        nargs="+",
        help="手写样本图片路径",
    )
    p_extract.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出",
    )

    # --- generate (default) ---
    p_gen = sub.add_parser("generate", help="分析样本并生成个性化原型字形包")
    p_gen.add_argument(
        "--sample",
        required=True,
        nargs="+",
        help="手写样本图片路径（支持多张）",
    )
    p_gen.add_argument(
        "--output_dir",
        required=True,
        help="输出目录（原型字形包保存位置）",
    )
    p_gen.add_argument(
        "--pack_name",
        default="personalized",
        help="原型字形包名称（默认 personalized）",
    )
    p_gen.add_argument(
        "--writer_id",
        default="personalized_user",
        help="书写者 ID（默认 personalized_user）",
    )
    p_gen.add_argument(
        "--charset",
        default=None,
        help="自定义字符集（字符串或文件路径）",
    )
    p_gen.add_argument(
        "--glyph_size",
        type=int,
        default=256,
        help="字形图片尺寸（默认 256）",
    )

    # Support top-level invocation without subcommand
    parser.add_argument(
        "--sample",
        dest="top_sample",
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output_dir",
        dest="top_output_dir",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pack_name",
        dest="top_pack_name",
        default="personalized",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--writer_id",
        dest="top_writer_id",
        default="personalized_user",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--charset",
        dest="top_charset",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--glyph_size",
        dest="top_glyph_size",
        type=int,
        default=256,
        help=argparse.SUPPRESS,
    )

    return parser


def _resolve_samples(sample_args: list[str]) -> list[Path]:
    paths = []
    for s in sample_args:
        p = Path(s)
        if not p.exists():
            print(f"错误: 样本文件不存在: {s}", file=sys.stderr)
            sys.exit(1)
        paths.append(p)
    return paths


def _load_charset(charset_arg: str | None) -> str | None:
    if charset_arg is None:
        return None
    p = Path(charset_arg)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return charset_arg


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run analyze subcommand."""
    from handwrite.personalization.sample_analyzer import SampleAnalyzer

    samples = _resolve_samples(args.samples)
    analyzer = SampleAnalyzer()

    results = []
    for path in samples:
        features = analyzer.analyze(path)
        results.append({"file": str(path), "features": features.__dict__})

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(f"\n=== {r['file']} ===")
            for k, v in r["features"].items():
                print(f"  {k}: {v}")


def cmd_extract(args: argparse.Namespace) -> None:
    """Run extract subcommand."""
    from handwrite.personalization.sample_analyzer import SampleAnalyzer
    from handwrite.personalization.style_extractor import StyleExtractor

    samples = _resolve_samples(args.samples)
    analyzer = SampleAnalyzer()
    extractor = StyleExtractor()

    from handwrite.personalization.style_extractor import StyleVector

    entries = []
    sv_list: list[StyleVector] = []
    for path in samples:
        features = analyzer.analyze(path)
        vector = extractor.extract(features)
        entries.append({"file": str(path), "style": vector.__dict__})
        sv_list.append(vector)

    avg = extractor.average_vectors(sv_list)

    output = {"samples": entries, "average_style": avg.__dict__}

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        for v in entries:
            print(f"\n--- {v['file']} ---")
            for k, val in v["style"].items():
                print(f"  {k}: {val}")
        print("\n=== 平均风格 ===")
        for k, val in avg.__dict__.items():
            print(f"  {k}: {val}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Run generate subcommand."""
    from handwrite.personalization.glyph_synthesizer import GlyphSynthesizer
    from handwrite.personalization.sample_analyzer import SampleAnalyzer
    from handwrite.personalization.style_extractor import StyleExtractor, StyleVector

    samples = _resolve_samples(args.sample)
    analyzer = SampleAnalyzer()
    extractor = StyleExtractor()

    print(f"分析 {len(samples)} 个手写样本...")
    vectors: list[StyleVector] = []
    for path in samples:
        features = analyzer.analyze(path)
        vector = extractor.extract(features)
        vectors.append(vector)
        print(f"  {path.name}: stroke_width={features.stroke_width_mean}, "
              f"slant={features.slant_angle}, connectivity={features.connectivity}")

    avg_style = extractor.average_vectors(vectors)
    print(f"\n平均风格向量:")
    for k, v in avg_style.__dict__.items():
        print(f"  {k}: {v}")

    charset = _load_charset(args.charset)
    print(f"\n开始合成字形包 -> {args.output_dir}")
    synthesizer = GlyphSynthesizer(
        avg_style, glyph_size=args.glyph_size
    )
    manifest_path = synthesizer.synthesize_pack(
        args.output_dir,
        charset=charset,
        pack_name=args.pack_name,
        writer_id=args.writer_id,
    )
    print(f"完成! manifest: {manifest_path}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Handle top-level --sample (no subcommand, treat as generate)
    if args.command is None:
        if args.top_sample:
            if not args.top_output_dir:
                print("错误: --output_dir 必须指定", file=sys.stderr)
                sys.exit(1)
            args.command = "generate"
            args.sample = args.top_sample
            args.output_dir = args.top_output_dir
            args.pack_name = args.top_pack_name
            args.writer_id = args.top_writer_id
            args.charset = args.top_charset
            args.glyph_size = args.top_glyph_size
        else:
            parser.print_help()
            sys.exit(0)

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
