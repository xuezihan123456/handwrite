from __future__ import annotations

import json
from pathlib import Path
import shutil
from contextlib import contextmanager
from uuid import uuid4
import zipfile

import pytest

import scripts.download_data as download_data


@contextmanager
def _project_temp_dir():
    root = Path(r"C:/Users/ASUS/.codex/memories/handwrite-download-tests")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_gnt(path: Path, payload: bytes = b"sample") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _make_split_dir(base_dir: Path, split: str, filenames: list[str]) -> Path:
    split_dir = base_dir / split
    for filename in filenames:
        _write_gnt(split_dir / filename, payload=filename.encode("ascii"))
    return split_dir


def _make_split_zip(
    base_dir: Path,
    split: str,
    filenames: list[str],
    *,
    include_root_dir: bool,
) -> Path:
    zip_path = base_dir / f"{split}.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for filename in filenames:
            arcname = f"{split}/{filename}" if include_root_dir else filename
            archive.writestr(arcname, filename.encode("ascii"))
    return zip_path


def test_discover_split_sources_recognizes_directories_and_zip_files() -> None:
    with _project_temp_dir() as tmp_path:
        scan_dir = tmp_path / "downloads"
        scan_dir.mkdir()
        split_dir = _make_split_dir(scan_dir, "HWDB1.0trn_gnt", ["001.gnt"])
        split_zip = _make_split_zip(
            scan_dir,
            "HWDB1.1tst_gnt",
            ["201.gnt", "202.gnt"],
            include_root_dir=True,
        )
        _make_split_dir(scan_dir, "unknown_split", ["999.gnt"])

        discovered = download_data.discover_split_sources(scan_dir)

        assert {split: candidate.kind for split, candidate in discovered.items()} == {
            "HWDB1.0trn_gnt": "directory",
            "HWDB1.1tst_gnt": "zip",
        }
        assert discovered["HWDB1.0trn_gnt"].path == split_dir
        assert discovered["HWDB1.1tst_gnt"].path == split_zip


def test_prepare_raw_data_copies_directories_and_extracts_zip_files() -> None:
    with _project_temp_dir() as tmp_path:
        scan_dir = tmp_path / "downloads"
        raw_dir = tmp_path / "data" / "raw"
        scan_dir.mkdir(parents=True)
        source_dir = _make_split_dir(
            scan_dir, "HWDB1.0trn_gnt", ["001.gnt", "nested/002.gnt"]
        )
        source_zip = _make_split_zip(
            scan_dir,
            "HWDB1.1tst_gnt",
            ["201.gnt", "nested/202.gnt"],
            include_root_dir=False,
        )

        summary = download_data.prepare_raw_data(scan_dir, raw_dir)

        assert (raw_dir / "HWDB1.0trn_gnt" / "001.gnt").read_bytes() == b"001.gnt"
        assert (raw_dir / "HWDB1.0trn_gnt" / "nested" / "002.gnt").read_bytes() == b"nested/002.gnt"
        assert (raw_dir / "HWDB1.1tst_gnt" / "201.gnt").read_bytes() == b"201.gnt"
        assert (raw_dir / "HWDB1.1tst_gnt" / "nested" / "202.gnt").read_bytes() == b"nested/202.gnt"
        assert source_dir.exists()
        assert source_zip.exists()

        assert summary["raw_dir"] == str(raw_dir.resolve())
        assert summary["splits"]["HWDB1.0trn_gnt"] == {
            "present": True,
            "source": str(source_dir.resolve()),
            "source_kind": "directory",
            "target_dir": str((raw_dir / "HWDB1.0trn_gnt").resolve()),
            "gnt_files": 2,
            "action": "copied",
        }
        assert summary["splits"]["HWDB1.1tst_gnt"] == {
            "present": True,
            "source": str(source_zip.resolve()),
            "source_kind": "zip",
            "target_dir": str((raw_dir / "HWDB1.1tst_gnt").resolve()),
            "gnt_files": 2,
            "action": "extracted",
        }
        assert summary["splits"]["HWDB1.0tst_gnt"]["present"] is False
        assert summary["splits"]["HWDB1.1trn_gnt"]["present"] is False


def test_prepare_raw_data_does_not_overwrite_existing_split_by_default() -> None:
    with _project_temp_dir() as tmp_path:
        scan_dir = tmp_path / "downloads"
        raw_dir = tmp_path / "data" / "raw"
        scan_dir.mkdir(parents=True)
        _make_split_dir(scan_dir, "HWDB1.0trn_gnt", ["new.gnt"])
        existing_dir = raw_dir / "HWDB1.0trn_gnt"
        _write_gnt(existing_dir / "existing.gnt", b"existing")

        summary = download_data.prepare_raw_data(scan_dir, raw_dir)

        assert (existing_dir / "existing.gnt").read_bytes() == b"existing"
        assert not (existing_dir / "new.gnt").exists()
        assert summary["splits"]["HWDB1.0trn_gnt"] == {
            "present": True,
            "source": str((scan_dir / "HWDB1.0trn_gnt").resolve()),
            "source_kind": "directory",
            "target_dir": str(existing_dir.resolve()),
            "gnt_files": 1,
            "action": "skipped_existing",
        }


def test_prepare_raw_data_overwrites_when_explicitly_enabled() -> None:
    with _project_temp_dir() as tmp_path:
        scan_dir = tmp_path / "downloads"
        raw_dir = tmp_path / "data" / "raw"
        scan_dir.mkdir(parents=True)
        _make_split_dir(scan_dir, "HWDB1.0trn_gnt", ["new.gnt"])
        existing_dir = raw_dir / "HWDB1.0trn_gnt"
        _write_gnt(existing_dir / "existing.gnt", b"existing")

        summary = download_data.prepare_raw_data(scan_dir, raw_dir, overwrite=True)

        assert not (existing_dir / "existing.gnt").exists()
        assert (existing_dir / "new.gnt").read_bytes() == b"new.gnt"
        assert summary["splits"]["HWDB1.0trn_gnt"]["action"] == "replaced"
        assert summary["splits"]["HWDB1.0trn_gnt"]["gnt_files"] == 1


def test_main_prints_structured_summary(capsys: pytest.CaptureFixture[str]) -> None:
    with _project_temp_dir() as tmp_path:
        scan_dir = tmp_path / "downloads"
        raw_dir = tmp_path / "data" / "raw"
        scan_dir.mkdir(parents=True)
        _make_split_zip(scan_dir, "HWDB1.0tst_gnt", ["101.gnt"], include_root_dir=True)

        exit_code = download_data.main(
            [
                "--scan_dir",
                str(scan_dir),
                "--raw_dir",
                str(raw_dir),
            ]
        )

        captured = capsys.readouterr()
        payload = json.loads(captured.out)

        assert exit_code == 0
        assert payload["raw_dir"] == str(raw_dir.resolve())
        assert payload["splits"]["HWDB1.0tst_gnt"] == {
            "present": True,
            "source": str((scan_dir / "HWDB1.0tst_gnt.zip").resolve()),
            "source_kind": "zip",
            "target_dir": str((raw_dir / "HWDB1.0tst_gnt").resolve()),
            "gnt_files": 1,
            "action": "extracted",
        }
        assert payload["splits"]["HWDB1.0trn_gnt"]["present"] is False
        assert captured.err == ""
