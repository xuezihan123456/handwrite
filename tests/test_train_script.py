from __future__ import annotations

import inspect
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_train_module():
    return importlib.import_module("scripts.train")


def _load_engine_train_module():
    return importlib.import_module("handwrite.engine.train")


def _install_training_fakes(
    monkeypatch: pytest.MonkeyPatch,
    train_module,
    calls: dict[str, object],
) -> None:
    def _make_image(fill_value: float) -> torch.Tensor:
        return torch.full((1, 256, 256), fill_value, dtype=torch.float32)

    def _collate_batch(samples: list[dict[str, object]]) -> dict[str, object]:
        return {
            "standard": torch.stack([sample["standard"] for sample in samples]),
            "handwrite": torch.stack([sample["handwrite"] for sample in samples]),
            "style_id": torch.tensor(
                [int(sample["style_id"]) for sample in samples],
                dtype=torch.long,
            ),
            "char": [str(sample["char"]) for sample in samples],
        }

    class FakeDataset:
        def __init__(
            self,
            metadata_path: Path,
            selected_styles_path: Path,
            transform=None,
        ) -> None:
            calls["dataset_args"] = {
                "metadata_path": metadata_path,
                "selected_styles_path": selected_styles_path,
                "transform": transform,
            }
            self.style_by_writer = {"writer_a": 0, "writer_b": 1}
            self.samples = [
                {
                    "standard_path": metadata_path.parent / "writer_a" / "4F60_standard.png",
                    "handwrite_path": metadata_path.parent / "writer_a" / "4F60_handwrite.png",
                    "style_id": 0,
                    "char": "你",
                },
                {
                    "standard_path": metadata_path.parent / "writer_b" / "597D_standard.png",
                    "handwrite_path": metadata_path.parent / "writer_b" / "597D_handwrite.png",
                    "style_id": 1,
                    "char": "好",
                },
            ]
            self._items = [
                {
                    "standard": _make_image(-1.0),
                    "handwrite": _make_image(-0.25),
                    "style_id": 0,
                    "char": "你",
                },
                {
                    "standard": _make_image(1.0),
                    "handwrite": _make_image(0.5),
                    "style_id": 1,
                    "char": "好",
                },
            ]

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, index: int) -> dict[str, object]:
            return self._items[index]

    class FakeDataLoader:
        def __init__(self, dataset, *, batch_size: int, shuffle: bool) -> None:
            calls["dataloader_instance"] = self
            calls["dataloader_args"] = {
                "dataset": dataset,
                "batch_size": batch_size,
                "shuffle": shuffle,
            }
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle

        def __iter__(self):
            batch: list[dict[str, object]] = []
            for index in range(len(self.dataset)):
                batch.append(self.dataset[index])
                if len(batch) == self.batch_size:
                    yield _collate_batch(batch)
                    batch = []
            if batch:
                yield _collate_batch(batch)

    class FakeGenerator:
        def __init__(self, *args, **kwargs) -> None:
            calls["generator_init"] = {"args": args, "kwargs": kwargs}
            calls["generator_instance"] = self
            self._parameters = [object()]
            calls["generator_parameters"] = self._parameters

        def parameters(self):
            return iter(self._parameters)

    class FakeDiscriminator:
        def __init__(self, *args, **kwargs) -> None:
            calls["discriminator_init"] = {"args": args, "kwargs": kwargs}
            calls["discriminator_instance"] = self
            self._parameters = [object()]
            calls["discriminator_parameters"] = self._parameters

        def parameters(self):
            return iter(self._parameters)

    class FakeAdam:
        def __init__(self, params, *, lr: float) -> None:
            parameter_list = list(params)
            optimizer_record = {
                "params": parameter_list,
                "lr": lr,
            }
            calls.setdefault("optimizer_inits", []).append(optimizer_record)

            if parameter_list == calls["generator_parameters"]:
                calls["generator_optimizer_instance"] = self
                calls["generator_optimizer_init"] = optimizer_record
            elif parameter_list == calls["discriminator_parameters"]:
                calls["discriminator_optimizer_instance"] = self
                calls["discriminator_optimizer_init"] = optimizer_record

    def fake_fit(
        *,
        generator,
        discriminator,
        dataloader,
        generator_optimizer,
        discriminator_optimizer,
        output_dir,
        epochs,
        l1_weight,
        label_shuffle_start,
        checkpoint_interval,
        sample_interval,
    ) -> None:
        assert output_dir.exists()
        assert output_dir.is_dir()
        calls["output_dir_ready_at_fit"] = True
        first_batch = next(iter(dataloader))
        calls["fit_first_batch"] = first_batch
        calls["fit_kwargs"] = {
            "generator": generator,
            "discriminator": discriminator,
            "dataloader": dataloader,
            "generator_optimizer": generator_optimizer,
            "discriminator_optimizer": discriminator_optimizer,
            "output_dir": output_dir,
            "epochs": epochs,
            "l1_weight": l1_weight,
            "label_shuffle_start": label_shuffle_start,
            "checkpoint_interval": checkpoint_interval,
            "sample_interval": sample_interval,
        }

    monkeypatch.setattr(train_module, "HandwriteDataset", FakeDataset, raising=False)
    monkeypatch.setattr(train_module, "DataLoader", FakeDataLoader, raising=False)
    monkeypatch.setattr(train_module, "Generator", FakeGenerator, raising=False)
    monkeypatch.setattr(train_module, "Discriminator", FakeDiscriminator, raising=False)
    monkeypatch.setattr(train_module, "Adam", FakeAdam, raising=False)
    monkeypatch.setattr(
        train_module,
        "engine_train",
        SimpleNamespace(fit=fake_fit),
        raising=False,
    )


def test_argument_parser_uses_expected_defaults(tmp_path: Path) -> None:
    train_module = _load_train_module()

    parser = train_module._build_argument_parser()
    args = parser.parse_args(
        [
            "--data_dir",
            str(tmp_path / "processed"),
            "--styles_file",
            str(tmp_path / "selected_styles.json"),
            "--output_dir",
            str(tmp_path / "weights"),
        ]
    )

    assert args.data_dir == tmp_path / "processed"
    assert args.styles_file == tmp_path / "selected_styles.json"
    assert args.output_dir == tmp_path / "weights"
    assert args.batch_size == 8
    assert args.epochs == 30
    assert args.lr == pytest.approx(0.0002)
    assert args.l1_weight == 100
    assert args.label_shuffle_start == 10
    assert args.save_interval == 2
    assert args.sample_interval == 500


def test_main_forwards_expected_training_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_module = _load_train_module()
    engine_train_module = _load_engine_train_module()
    calls: dict[str, object] = {}
    _install_training_fakes(monkeypatch, train_module, calls)

    data_dir = tmp_path / "processed"
    styles_file = tmp_path / "selected_styles.json"
    output_dir = tmp_path / "weights"
    data_dir.mkdir()
    styles_file.write_text("{}", encoding="utf-8")

    exit_code = train_module.main(
        [
            "--data_dir",
            str(data_dir),
            "--styles_file",
            str(styles_file),
            "--output_dir",
            str(output_dir),
            "--batch_size",
            "4",
            "--epochs",
            "12",
            "--lr",
            "0.001",
            "--l1_weight",
            "75",
            "--label_shuffle_start",
            "3",
            "--save_interval",
            "5",
            "--sample_interval",
            "120",
        ]
    )

    assert exit_code == 0
    assert calls["dataset_args"] == {
        "metadata_path": data_dir / "metadata.json",
        "selected_styles_path": styles_file,
        "transform": None,
    }
    assert calls["dataloader_args"]["batch_size"] == 4
    assert calls["dataloader_args"]["shuffle"] is True
    assert calls["generator_init"]["kwargs"] == {"num_styles": 2}
    assert calls["discriminator_init"]["kwargs"] == {"num_styles": 2}
    assert calls["generator_optimizer_init"]["lr"] == pytest.approx(0.001)
    assert calls["discriminator_optimizer_init"]["lr"] == pytest.approx(0.001)
    assert calls["fit_kwargs"]["dataloader"] is calls["dataloader_instance"]
    assert calls["fit_kwargs"]["generator"] is calls["generator_instance"]
    assert calls["fit_kwargs"]["discriminator"] is calls["discriminator_instance"]
    assert (
        calls["fit_kwargs"]["generator_optimizer"]
        is calls["generator_optimizer_instance"]
    )
    assert (
        calls["fit_kwargs"]["discriminator_optimizer"]
        is calls["discriminator_optimizer_instance"]
    )
    assert calls["fit_kwargs"]["output_dir"] == output_dir
    assert calls["fit_kwargs"]["epochs"] == 12
    assert calls["fit_kwargs"]["l1_weight"] == 75
    assert calls["fit_kwargs"]["label_shuffle_start"] == 3
    assert calls["fit_kwargs"]["checkpoint_interval"] == 5
    assert calls["fit_kwargs"]["sample_interval"] == 120
    real_fit_parameters = inspect.signature(engine_train_module.fit).parameters
    assert "label_shuffle_start" in real_fit_parameters
    assert set(calls["fit_kwargs"]) <= set(real_fit_parameters)
    fit_first_batch = calls["fit_first_batch"]
    assert set(fit_first_batch) == {"standard", "handwrite", "style_id", "char"}
    assert tuple(fit_first_batch["standard"].shape) == (2, 1, 256, 256)
    assert tuple(fit_first_batch["handwrite"].shape) == (2, 1, 256, 256)
    assert torch.equal(fit_first_batch["style_id"], torch.tensor([0, 1]))
    assert fit_first_batch["char"] == ["你", "好"]
    assert set(calls["fit_kwargs"]) == {
        "generator",
        "discriminator",
        "dataloader",
        "generator_optimizer",
        "discriminator_optimizer",
        "output_dir",
        "epochs",
        "l1_weight",
        "label_shuffle_start",
        "checkpoint_interval",
        "sample_interval",
    }


def test_main_creates_output_dir_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_module = _load_train_module()
    calls: dict[str, object] = {}
    _install_training_fakes(monkeypatch, train_module, calls)

    data_dir = tmp_path / "processed"
    styles_file = tmp_path / "selected_styles.json"
    output_dir = tmp_path / "weights"
    data_dir.mkdir()
    styles_file.write_text("{}", encoding="utf-8")

    exit_code = train_module.main(
        [
            "--data_dir",
            str(data_dir),
            "--styles_file",
            str(styles_file),
            "--output_dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert calls["output_dir_ready_at_fit"] is True
    assert calls["fit_kwargs"]["output_dir"] == output_dir
