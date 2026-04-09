import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.preprocess import build_processed_dataset
from handwrite.data.dataset import HandwriteDataset


def _write_sample_image(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (256, 256), color=color).save(path)


def test_dataset_filters_to_selected_styles_and_normalizes_images(tmp_path: Path) -> None:
    standard_a = tmp_path / "writer_001" / "4F60_standard.png"
    handwrite_a = tmp_path / "writer_001" / "4F60_handwrite.png"
    standard_b = tmp_path / "writer_002" / "597D_standard.png"
    handwrite_b = tmp_path / "writer_002" / "597D_handwrite.png"
    ignored_standard = tmp_path / "writer_999" / "6D4B_standard.png"
    ignored_handwrite = tmp_path / "writer_999" / "6D4B_handwrite.png"

    _write_sample_image(standard_a, 255)
    _write_sample_image(handwrite_a, 0)
    _write_sample_image(standard_b, 255)
    _write_sample_image(handwrite_b, 32)
    _write_sample_image(ignored_standard, 255)
    _write_sample_image(ignored_handwrite, 64)

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "writer_id": "001",
                        "char": "你",
                        "standard": "writer_001/4F60_standard.png",
                        "handwrite": "writer_001/4F60_handwrite.png",
                    },
                    {
                        "writer_id": "002",
                        "char": "好",
                        "standard": "writer_002/597D_standard.png",
                        "handwrite": "writer_002/597D_handwrite.png",
                    },
                    {
                        "writer_id": "999",
                        "char": "测",
                        "standard": "writer_999/6D4B_standard.png",
                        "handwrite": "writer_999/6D4B_handwrite.png",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    styles_path = tmp_path / "selected_styles.json"
    styles_path.write_text(
        json.dumps(
            {
                "styles": [
                    {"id": 0, "name": "工整楷书", "writer_id": "001"},
                    {"id": 1, "name": "圆润可爱", "writer_id": "002"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dataset = HandwriteDataset(str(metadata_path), str(styles_path))

    assert len(dataset) == 2

    first = dataset[0]
    assert set(first) == {"standard", "handwrite", "style_id", "char"}
    assert tuple(first["standard"].shape) == (1, 256, 256)
    assert tuple(first["handwrite"].shape) == (1, 256, 256)
    assert float(first["standard"].max()) <= 1.0
    assert float(first["standard"].min()) >= -1.0
    assert first["style_id"] in {0, 1}
    assert first["char"] in {"你", "好"}


def test_dataset_can_read_metadata_generated_by_preprocess_pipeline(tmp_path: Path) -> None:
    def find_chinese_font() -> str:
        candidates = [
            Path("C:/Windows/Fonts/NotoSerifSC-VF.ttf"),
            Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simsun.ttc"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError("No Chinese-capable font found in expected Windows font paths.")

    bitmap = np.full((64, 64), 255, dtype=np.uint8)
    bitmap[16:48, 20:40] = 0
    output_dir = tmp_path / "processed"

    build_processed_dataset(
        {"001": [("你", bitmap), ("好", bitmap)]},
        output_dir=output_dir,
        font_path=find_chinese_font(),
        charset=["你", "好"],
        min_writer_coverage=1.0,
    )

    styles_path = tmp_path / "selected_styles.json"
    styles_path.write_text(
        json.dumps(
            {"styles": [{"id": 0, "name": "工整楷书", "writer_id": "001"}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dataset = HandwriteDataset(str(output_dir / "metadata.json"), str(styles_path))

    assert len(dataset) == 2
    assert dataset[0]["char"] in {"你", "好"}
