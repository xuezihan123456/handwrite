"""Dataset definitions for handwriting training."""

from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class HandwriteDataset(Dataset):
    """Load paired standard/handwrite images for selected styles."""

    def __init__(
        self,
        metadata_path: str,
        selected_styles_path: str,
        transform=None,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.selected_styles_path = Path(selected_styles_path)
        self.transform = transform

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        styles = json.loads(self.selected_styles_path.read_text(encoding="utf-8"))
        self.style_by_writer = {
            entry["writer_id"]: entry["id"] for entry in styles.get("styles", [])
        }

        self.samples: list[dict[str, object]] = []
        for pair in metadata.get("pairs", []):
            writer_id = pair["writer_id"]
            if writer_id not in self.style_by_writer:
                continue

            self.samples.append(
                {
                    "standard_path": self._resolve_path(pair["standard"]),
                    "handwrite_path": self._resolve_path(pair["handwrite"]),
                    "style_id": self.style_by_writer[writer_id],
                    "char": pair["char"],
                }
            )

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path

        metadata_relative = (self.metadata_path.parent / path).resolve()
        if metadata_relative.exists():
            return metadata_relative

        project_relative = (Path.cwd() / path).resolve()
        if project_relative.exists():
            return project_relative

        return metadata_relative

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        image = Image.open(path).convert("L")
        if image.size != (256, 256):
            image = image.resize((256, 256))
        return image

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(array).unsqueeze(0)
        return (tensor / 127.5) - 1.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        standard = self._load_image(sample["standard_path"])
        handwrite = self._load_image(sample["handwrite_path"])

        if self.transform is not None:
            handwrite = self.transform(handwrite)

        return {
            "standard": self._to_tensor(standard),
            "handwrite": self._to_tensor(handwrite),
            "style_id": sample["style_id"],
            "char": sample["char"],
        }
