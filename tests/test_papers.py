"""Tests for the paper template ecosystem."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from handwrite.papers.builtin_papers import BUILTIN_PAPER_DEFS, builtin_paper_names
from handwrite.papers.custom_paper import load_paper_image, load_paper_json, save_paper_json
from handwrite.papers.paper_registry import PaperRegistry, get_paper, list_papers
from handwrite.papers.paper_renderer import render_paper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PAPERS_DIR = PROJECT_ROOT / "data" / "papers"

EXPECTED_PAPERS = [
    "康奈尔笔记",
    "作文稿纸",
    "五线谱",
    "错题本",
    "思维导图",
    "英语练习纸",
]


# ---------------------------------------------------------------------------
# Builtin papers
# ---------------------------------------------------------------------------


class TestBuiltinPapers:
    def test_all_builtin_names_present(self) -> None:
        names = builtin_paper_names()
        for name in EXPECTED_PAPERS:
            assert name in names, f"Missing built-in paper: {name}"

    def test_all_builtin_defs_have_required_keys(self) -> None:
        for name, defn in BUILTIN_PAPER_DEFS.items():
            assert "name" in defn, f"{name} missing 'name'"
            assert "size" in defn, f"{name} missing 'size'"
            assert "regions" in defn, f"{name} missing 'regions'"
            assert defn["name"] == name

    def test_all_builtin_sizes_are_a4(self) -> None:
        for name, defn in BUILTIN_PAPER_DEFS.items():
            assert defn["size"] == [2480, 3508], f"{name} is not A4"


# ---------------------------------------------------------------------------
# JSON data files
# ---------------------------------------------------------------------------


class TestJsonDataFiles:
    @pytest.mark.parametrize("paper_name", EXPECTED_PAPERS)
    def test_json_file_exists_and_valid(self, paper_name: str) -> None:
        json_files = list(DATA_PAPERS_DIR.glob("*.json"))
        names_in_dir = []
        for jf in json_files:
            data = json.loads(jf.read_text(encoding="utf-8"))
            names_in_dir.append(data.get("name", jf.stem))
        assert paper_name in names_in_dir, (
            f"Paper '{paper_name}' not found in {DATA_PAPERS_DIR}"
        )

    def test_all_json_files_parse(self) -> None:
        for jf in sorted(DATA_PAPERS_DIR.glob("*.json")):
            data = json.loads(jf.read_text(encoding="utf-8"))
            assert "name" in data, f"{jf.name} missing 'name'"
            assert "regions" in data, f"{jf.name} missing 'regions'"


# ---------------------------------------------------------------------------
# PaperRegistry
# ---------------------------------------------------------------------------


class TestPaperRegistry:
    def test_registry_loads_all_papers(self) -> None:
        registry = PaperRegistry()
        names = registry.list_names()
        for name in EXPECTED_PAPERS:
            assert name in names, f"Registry missing paper: {name}"

    def test_registry_get_returns_definition(self) -> None:
        registry = PaperRegistry()
        defn = registry.get("康奈尔笔记")
        assert defn is not None
        assert defn["name"] == "康奈尔笔记"
        assert defn["size"] == [2480, 3508]

    def test_registry_get_missing_returns_none(self) -> None:
        registry = PaperRegistry()
        assert registry.get("不存在的纸张") is None

    def test_registry_contains(self) -> None:
        registry = PaperRegistry()
        assert "康奈尔笔记" in registry
        assert "不存在" not in registry

    def test_registry_len(self) -> None:
        registry = PaperRegistry()
        assert len(registry) >= len(EXPECTED_PAPERS)

    def test_registry_register_custom(self, tmp_path: Path) -> None:
        registry = PaperRegistry()
        custom = {
            "name": "自定义测试纸",
            "size": [2480, 3508],
            "regions": [
                {"type": "line", "x1": 0, "y1": 100, "x2": 2480, "y2": 100, "color": 128}
            ],
        }
        registry.register(custom)
        assert "自定义测试纸" in registry
        defn = registry.get("自定义测试纸")
        assert defn is not None
        assert len(defn["regions"]) == 1

    def test_registry_register_no_name_raises(self) -> None:
        registry = PaperRegistry()
        with pytest.raises(ValueError, match="name"):
            registry.register({"size": [2480, 3508], "regions": []})

    def test_registry_load_json(self, tmp_path: Path) -> None:
        json_path = tmp_path / "test_paper.json"
        json_path.write_text(
            json.dumps(
                {
                    "name": "测试JSON纸张",
                    "size": [2480, 3508],
                    "regions": [
                        {"type": "line", "x1": 0, "y1": 500, "x2": 2480, "y2": 500, "color": 200}
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        registry = PaperRegistry()
        name = registry.load_json(json_path)
        assert name == "测试JSON纸张"
        assert "测试JSON纸张" in registry


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    def test_list_papers(self) -> None:
        names = list_papers()
        assert isinstance(names, list)
        for name in EXPECTED_PAPERS:
            assert name in names

    def test_get_paper(self) -> None:
        defn = get_paper("五线谱")
        assert defn is not None
        assert defn["name"] == "五线谱"

    def test_get_paper_missing(self) -> None:
        assert get_paper("不存在") is None


# ---------------------------------------------------------------------------
# Paper renderer
# ---------------------------------------------------------------------------


class TestPaperRenderer:
    @pytest.mark.parametrize("paper_name", EXPECTED_PAPERS)
    def test_render_returns_grayscale_image(self, paper_name: str) -> None:
        defn = get_paper(paper_name)
        assert defn is not None, f"Paper '{paper_name}' not found"
        img = render_paper(defn)
        assert isinstance(img, Image.Image)
        assert img.mode == "L"
        assert img.size == (2480, 3508)

    def test_render_all_builtins(self) -> None:
        for name in EXPECTED_PAPERS:
            defn = get_paper(name)
            img = render_paper(defn)
            assert img.size == (2480, 3508), f"{name} wrong size"

    def test_render_empty_definition(self) -> None:
        img = render_paper({"size": [800, 600], "regions": []})
        assert img.size == (800, 600)
        # All white
        assert img.getpixel((0, 0)) == 255

    def test_render_single_line(self) -> None:
        defn = {
            "size": [100, 100],
            "regions": [
                {"type": "line", "x1": 0, "y1": 50, "x2": 100, "y2": 50, "color": 0}
            ],
        }
        img = render_paper(defn)
        # Line at y=50 should be dark
        assert img.getpixel((50, 50)) == 0
        # Background should be white
        assert img.getpixel((50, 10)) == 255

    def test_render_ellipse(self) -> None:
        defn = {
            "size": [200, 200],
            "regions": [
                {"type": "ellipse", "cx": 100, "cy": 100, "rx": 50, "ry": 30, "color": 100, "width": 2}
            ],
        }
        img = render_paper(defn)
        assert img.size == (200, 200)

    def test_render_default_size_fallback(self) -> None:
        img = render_paper({"regions": []})
        assert img.size == (2480, 3508)


# ---------------------------------------------------------------------------
# Custom paper loader
# ---------------------------------------------------------------------------


class TestCustomPaperLoader:
    def test_load_paper_json(self, tmp_path: Path) -> None:
        json_path = tmp_path / "custom.json"
        json_path.write_text(
            json.dumps(
                {
                    "name": "我的纸张",
                    "size": [2480, 3508],
                    "regions": [
                        {"type": "line", "x1": 0, "y1": 100, "x2": 2480, "y2": 100, "color": 200}
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        defn = load_paper_json(json_path)
        assert defn["name"] == "我的纸张"
        assert defn["size"] == [2480, 3508]
        assert len(defn["regions"]) == 1

    def test_load_paper_json_defaults(self, tmp_path: Path) -> None:
        json_path = tmp_path / "minimal.json"
        json_path.write_text("{}", encoding="utf-8")
        defn = load_paper_json(json_path)
        assert defn["name"] == "minimal"
        assert defn["size"] == [2480, 3508]
        assert defn["regions"] == []

    def test_load_paper_image(self, tmp_path: Path) -> None:
        img_path = tmp_path / "bg.png"
        Image.new("L", (800, 600), color=200).save(str(img_path))
        defn = load_paper_image(img_path, name="背景纸")
        assert defn["name"] == "背景纸"
        assert defn["size"] == [800, 600]
        assert "_image" in defn

    def test_save_paper_json(self, tmp_path: Path) -> None:
        defn = {
            "name": "可保存纸张",
            "size": [2480, 3508],
            "regions": [{"type": "line", "x1": 0, "y1": 100, "x2": 2480, "y2": 100}],
            "_source": "/internal/path",
        }
        out_path = tmp_path / "saved.json"
        save_paper_json(defn, out_path)
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["name"] == "可保存纸张"
        assert "_source" not in loaded
        assert "_image" not in loaded
