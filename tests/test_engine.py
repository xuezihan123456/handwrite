from pathlib import Path

import handwrite.engine.model as engine_model
from PIL import Image
import pytest
import torch

from handwrite.engine.generator import Generator
from handwrite.engine.model import StyleEngine
from handwrite.prototypes import load_builtin_prototype_library, load_prototype_library


def _assert_visible_grayscale_image(image: Image.Image) -> None:
    assert isinstance(image, Image.Image)
    assert image.mode == "L"
    assert image.size == (256, 256)
    assert image.point(lambda value: 255 - value).getbbox() is not None


def _disable_default_font_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        engine_model,
        "_DEFAULT_FONT_CANDIDATES",
        (
            tmp_path / "missing-font-1.ttf",
            tmp_path / "missing-font-2.ttf",
        ),
    )


def _disable_default_weight_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        engine_model,
        "_DEFAULT_WEIGHTS_CANDIDATES",
        (
            tmp_path / "missing-generator-1.pt",
            tmp_path / "missing-generator-2.pth",
            tmp_path / "missing-checkpoint-1.pt",
            tmp_path / "missing-checkpoint-2.pth",
        ),
        raising=False,
    )


def _disable_default_runtime_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_default_font_candidates(monkeypatch, tmp_path)
    _disable_default_weight_candidates(monkeypatch, tmp_path)


def _save_real_generator_weights(
    tmp_path,
    *,
    num_styles: int,
    as_checkpoint: bool,
    filename: str,
) -> tuple[torch.Tensor, Path]:
    generator = Generator(num_styles=num_styles)
    expected_embedding = torch.arange(
        num_styles * generator.style_embedding.embedding_dim,
        dtype=generator.style_embedding.weight.dtype,
    ).view(num_styles, generator.style_embedding.embedding_dim)

    with torch.no_grad():
        generator.style_embedding.weight.copy_(expected_embedding)

    weights_path = tmp_path / filename
    payload = (
        {"generator_state_dict": generator.state_dict()}
        if as_checkpoint
        else generator.state_dict()
    )
    torch.save(payload, weights_path)
    return expected_embedding, weights_path


def _write_custom_prototype_pack(tmp_path: Path, *, char: str = "龘") -> Path:
    pack_dir = tmp_path / "custom-pack"
    glyph_dir = pack_dir / "glyphs"
    glyph_dir.mkdir(parents=True, exist_ok=True)
    glyph_path = glyph_dir / f"U{ord(char):04X}.png"
    image = Image.new("L", (256, 256), color=255)
    image.paste(32, (88, 88, 168, 168))
    image.save(glyph_path)
    manifest_path = pack_dir / "manifest.json"
    manifest_path.write_text(
        (
            "{\n"
            '  "name": "custom-pack",\n'
            '  "glyphs": [\n'
            f'    {{"char": "{char}", "file": "glyphs/U{ord(char):04X}.png", "writer_id": "custom"}}\n'
            "  ]\n"
            "}\n"
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_style_engine_constructs_and_generates_without_default_fonts(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_default_runtime_candidates(monkeypatch, tmp_path)
    engine = StyleEngine()

    image = engine.generate_char("\u4f60", 0)

    _assert_visible_grayscale_image(image)


@pytest.mark.parametrize("as_checkpoint", [False, True], ids=["state-dict", "checkpoint"])
def test_style_engine_loads_real_generator_from_explicit_weights_path_with_inferred_style_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path, as_checkpoint: bool
) -> None:
    _disable_default_font_candidates(monkeypatch, tmp_path)
    expected_embedding, weights_path = _save_real_generator_weights(
        tmp_path,
        num_styles=7,
        as_checkpoint=as_checkpoint,
        filename="generator-checkpoint.pt" if as_checkpoint else "generator-state-dict.pt",
    )

    engine = StyleEngine(weights_path=weights_path, supported_chars={"\u4f60"})

    assert isinstance(engine._generator, Generator)
    assert engine._generator.style_embedding.num_embeddings == 7
    assert torch.equal(engine._generator.style_embedding.weight, expected_embedding)


def test_style_engine_discovers_latest_epoch_checkpoint_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_default_font_candidates(monkeypatch, tmp_path)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    _save_real_generator_weights(
        weights_dir,
        num_styles=6,
        as_checkpoint=True,
        filename="checkpoint_epoch_2.pt",
    )
    expected_embedding, _ = _save_real_generator_weights(
        weights_dir,
        num_styles=8,
        as_checkpoint=True,
        filename="checkpoint_epoch_10.pt",
    )
    monkeypatch.setattr(
        engine_model,
        "_DEFAULT_WEIGHTS_CANDIDATES",
        tuple(weights_dir / name for name in ("generator.pt", "generator.pth", "checkpoint.pt", "checkpoint.pth")),
        raising=False,
    )

    engine = StyleEngine(supported_chars={"\u4f60"})

    assert isinstance(engine._generator, Generator)
    assert engine._generator.style_embedding.num_embeddings == 8
    assert torch.equal(engine._generator.style_embedding.weight, expected_embedding)


def test_style_engine_prefers_discovered_better_checkpoint_over_fixed_name_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_default_font_candidates(monkeypatch, tmp_path)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    _save_real_generator_weights(
        weights_dir,
        num_styles=4,
        as_checkpoint=True,
        filename="checkpoint.pt",
    )
    expected_embedding, _ = _save_real_generator_weights(
        weights_dir,
        num_styles=9,
        as_checkpoint=True,
        filename="checkpoint_epoch_10.pt",
    )
    monkeypatch.setattr(
        engine_model,
        "_DEFAULT_WEIGHTS_CANDIDATES",
        tuple(
            weights_dir / name
            for name in ("generator.pt", "generator.pth", "checkpoint.pt", "checkpoint.pth")
        ),
        raising=False,
    )

    engine = StyleEngine(supported_chars={"\u4f60"})

    assert isinstance(engine._generator, Generator)
    assert engine._generator.style_embedding.num_embeddings == 9
    assert torch.equal(engine._generator.style_embedding.weight, expected_embedding)


def test_style_engine_prefers_epoch_checkpoint_over_fixed_generator_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_default_font_candidates(monkeypatch, tmp_path)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    _save_real_generator_weights(
        weights_dir,
        num_styles=4,
        as_checkpoint=False,
        filename="generator.pt",
    )
    expected_embedding, _ = _save_real_generator_weights(
        weights_dir,
        num_styles=9,
        as_checkpoint=True,
        filename="checkpoint_epoch_10.pt",
    )
    monkeypatch.setattr(
        engine_model,
        "_DEFAULT_WEIGHTS_CANDIDATES",
        tuple(
            weights_dir / name
            for name in ("generator.pt", "generator.pth", "checkpoint.pt", "checkpoint.pth")
        ),
        raising=False,
    )

    engine = StyleEngine(supported_chars={"\u4f60"})

    assert isinstance(engine._generator, Generator)
    assert engine._generator.style_embedding.num_embeddings == 9
    assert torch.equal(engine._generator.style_embedding.weight, expected_embedding)


@pytest.mark.parametrize(
    "filename",
    ["generator_state_dict.pth", "best_generator.pt"],
    ids=["generator-state-dict", "best-generator"],
)
def test_style_engine_discovers_common_weight_filenames_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path, filename: str
) -> None:
    _disable_default_font_candidates(monkeypatch, tmp_path)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    expected_embedding, _ = _save_real_generator_weights(
        weights_dir,
        num_styles=7,
        as_checkpoint=False,
        filename=filename,
    )
    monkeypatch.setattr(
        engine_model,
        "_DEFAULT_WEIGHTS_CANDIDATES",
        tuple(weights_dir / name for name in ("generator.pt", "generator.pth", "checkpoint.pt", "checkpoint.pth")),
        raising=False,
    )

    engine = StyleEngine(supported_chars={"\u4f60"})

    assert isinstance(engine._generator, Generator)
    assert engine._generator.style_embedding.num_embeddings == 7
    assert torch.equal(engine._generator.style_embedding.weight, expected_embedding)


def test_generate_char_uses_supported_char_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_default_runtime_candidates(monkeypatch, tmp_path)
    engine = StyleEngine(supported_chars={"\u4f60"})
    supported_image = Image.new("L", (256, 256), color=192)
    calls: list[tuple[str, int]] = []

    def fake_gan_generate(char: str, style_id: int) -> Image.Image:
        calls.append((char, style_id))
        return supported_image

    monkeypatch.setattr(engine, "_gan_generate", fake_gan_generate, raising=False)
    monkeypatch.setattr(
        engine,
        "_fallback_render",
        lambda char: pytest.fail(f"Unexpected fallback for supported char: {char}"),
        raising=False,
    )

    image = engine.generate_char("\u4f60", 2)

    assert image is supported_image
    assert calls == [("\u4f60", 2)]


def test_generate_char_respects_explicitly_empty_supported_chars(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_default_runtime_candidates(monkeypatch, tmp_path)
    engine = StyleEngine(supported_chars=set())
    engine.prototype_library = None
    fallback_image = Image.new("L", (256, 256), color=160)

    monkeypatch.setattr(
        engine,
        "_gan_generate",
        lambda char, style_id: pytest.fail(
            f"Unexpected supported-char path for empty supported_chars: {char}/{style_id}"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        engine,
        "_fallback_render",
        lambda char: fallback_image,
        raising=False,
    )

    image = engine.generate_char("\u4f60", 1)

    assert image is fallback_image

def test_generate_char_prefers_prototype_glyph_when_model_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_default_runtime_candidates(monkeypatch, tmp_path)
    prototype_library = load_builtin_prototype_library()
    engine = StyleEngine(
        supported_chars={"学"},
        prototype_library=prototype_library,
    )

    generated = engine.generate_char("你", 2)
    prototype = prototype_library.get_glyph_image("你")

    assert generated.tobytes() == prototype.tobytes()


def test_generate_char_avoids_default_note_prototypes_for_non_default_styles(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_default_runtime_candidates(monkeypatch, tmp_path)
    prototype_library = load_builtin_prototype_library()
    engine = StyleEngine(
        supported_chars={"学"},
        prototype_library=prototype_library,
    )

    generated = engine.generate_char("你", 0)
    prototype = prototype_library.get_glyph_image("你")

    assert generated.tobytes() != prototype.tobytes()


def test_generate_char_uses_custom_prototype_pack_for_note_style(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_default_runtime_candidates(monkeypatch, tmp_path)
    manifest_path = _write_custom_prototype_pack(tmp_path)
    engine = StyleEngine(
        supported_chars={"学"},
        prototype_pack=manifest_path,
    )

    generated = engine.generate_char("龘", 2)
    prototype = load_prototype_library(manifest_path).get_glyph_image("龘")

    assert generated.tobytes() == prototype.tobytes()
    assert engine.prototype_library is not None
    assert engine.prototype_library.name == "custom-pack"


def test_inspect_char_reports_fallback_for_low_realism_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_default_runtime_candidates(monkeypatch, tmp_path)
    engine = StyleEngine(
        supported_chars={"学"},
        prototype_library=load_builtin_prototype_library(),
    )

    inspection = engine.inspect_char("龘", 0)

    assert inspection["route"] == "fallback"
    assert inspection["is_low_realism"] is True
    assert inspection["char"] == "龘"


def test_inspect_text_reports_prototype_and_fallback_routes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_default_runtime_candidates(monkeypatch, tmp_path)
    engine = StyleEngine(
        supported_chars={"学"},
        prototype_library=load_builtin_prototype_library(),
    )
    monkeypatch.setattr(engine, "_generator", object(), raising=False)

    report = engine.inspect_text("你学龘", 2)

    assert report["prototype_covered_characters"] == ["你"]
    assert report["model_supported_characters"] == ["学"]
    assert report["fallback_characters"] == ["龘"]
    assert report["style"] == "行书流畅"
    assert report["total_characters"] == 3


def test_inspect_text_reports_custom_prototype_pack_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_default_runtime_candidates(monkeypatch, tmp_path)
    manifest_path = _write_custom_prototype_pack(tmp_path)
    engine = StyleEngine(
        supported_chars={"学"},
        prototype_pack=manifest_path.parent,
    )

    report = engine.inspect_text("龘学", 2)

    assert report["prototype_pack_name"] == "custom-pack"
    assert report["prototype_source"] == str(manifest_path.resolve())
    assert report["prototype_source_kind"] == "custom"
    assert report["prototype_covered_characters"] == ["龘"]
    assert report["model_supported_characters"] == []

def test_generate_char_rejects_style_ids_not_supported_by_loaded_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _disable_default_font_candidates(monkeypatch, tmp_path)
    _, weights_path = _save_real_generator_weights(
        tmp_path,
        num_styles=3,
        as_checkpoint=True,
        filename="checkpoint_epoch_1.pt",
    )

    engine = StyleEngine(weights_path=weights_path, supported_chars={"\u4f60"})

    assert isinstance(engine._generator, Generator)
    assert engine._generator.style_embedding.num_embeddings == 3
    with pytest.raises(ValueError, match="Unsupported style_id: 4"):
        engine.generate_char("\u4f60", 4)


@pytest.mark.parametrize("broken_weights", [False, True], ids=["missing", "unloadable"])
def test_style_engine_degrades_cleanly_when_weights_are_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path, broken_weights: bool
) -> None:
    _disable_default_runtime_candidates(monkeypatch, tmp_path)

    weights_path = tmp_path / "generator.pt"
    if broken_weights:
        weights_path.write_text("not a torch checkpoint", encoding="utf-8")

    engine = StyleEngine(weights_path=weights_path, supported_chars={"\u4f60"})
    fallback_engine = StyleEngine(supported_chars={"\u4f60"})

    image = engine.generate_char("\u4f60", 2)
    fallback_image = fallback_engine.generate_char("\u4f60", 2)

    assert image.mode == fallback_image.mode
    assert image.size == fallback_image.size
    assert image.tobytes() == fallback_image.tobytes()


def test_generate_char_rejects_invalid_style_ids() -> None:
    engine = StyleEngine(font_path=None, weights_path="missing-generator.pt", supported_chars={"\u4f60"})

    with pytest.raises(ValueError, match="Unsupported style_id: 99"):
        engine.generate_char("\u4f60", 99)
