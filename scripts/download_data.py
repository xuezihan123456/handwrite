"""Prepare local CASIA/HWDB archives into the raw data layout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Mapping
import zipfile


SUPPORTED_SPLITS = (
    "HWDB1.0trn_gnt",
    "HWDB1.0tst_gnt",
    "HWDB1.1trn_gnt",
    "HWDB1.1tst_gnt",
)

SOURCE_PRIORITY = {"directory": 0, "zip": 1}


@dataclass(frozen=True)
class SplitSource:
    """A discovered local source for a supported CASIA/HWDB split."""

    split: str
    path: Path
    kind: str


def identify_split_name(path: str | Path) -> str | None:
    """Return the supported split name encoded in a path name, if any."""
    candidate_path = Path(path)
    if candidate_path.suffix.lower() == ".zip":
        candidate_name = candidate_path.stem.lower()
    else:
        candidate_name = candidate_path.name.lower()

    for split in SUPPORTED_SPLITS:
        if split.lower() in candidate_name:
            return split
    return None


def _get_source_kind(path: Path) -> str | None:
    if path.is_dir():
        return "directory"
    if path.is_file() and path.suffix.lower() == ".zip":
        return "zip"
    return None


def _prefer_source(current: SplitSource | None, candidate: SplitSource) -> SplitSource:
    if current is None:
        return candidate
    current_priority = SOURCE_PRIORITY.get(current.kind, 99)
    candidate_priority = SOURCE_PRIORITY.get(candidate.kind, 99)
    if candidate_priority < current_priority:
        return candidate
    if candidate_priority > current_priority:
        return current
    return min(current, candidate, key=lambda source: source.path.name.lower())


def discover_split_sources(scan_dir: str | Path) -> dict[str, SplitSource]:
    """Scan a directory for supported CASIA/HWDB zip files or split directories."""
    source_dir = Path(scan_dir)
    if not source_dir.exists():
        return {}

    discovered: dict[str, SplitSource] = {}
    for path in sorted(source_dir.iterdir(), key=lambda item: item.name.lower()):
        split = identify_split_name(path)
        kind = _get_source_kind(path)
        if split is None or kind is None:
            continue
        candidate = SplitSource(split=split, path=path.resolve(), kind=kind)
        discovered[split] = _prefer_source(discovered.get(split), candidate)
    return discovered


def count_gnt_files(split_dir: str | Path) -> int:
    """Count `.gnt` files under a split directory recursively."""
    directory = Path(split_dir)
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".gnt")


def _member_relative_path(member_path: PurePosixPath, split: str) -> Path | None:
    split_index = next(
        (index for index, part in enumerate(member_path.parts) if identify_split_name(part) == split),
        None,
    )
    if split_index is None:
        relative_parts = member_path.parts
    else:
        relative_parts = member_path.parts[split_index + 1 :]

    if not relative_parts:
        return None
    return Path(*relative_parts)


def _safe_extract_archive(archive: zipfile.ZipFile, destination_dir: Path, split: str) -> None:
    for member in archive.infolist():
        if member.is_dir():
            continue
        member_path = PurePosixPath(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Archive member escapes extraction root: {member.filename}")

        relative_path = _member_relative_path(member_path, split)
        if relative_path is None:
            continue

        target_path = destination_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source_handle, target_path.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)


def _copy_split_directory(source_dir: Path, target_dir: Path) -> None:
    shutil.copytree(source_dir, target_dir)


def _extract_split_archive(archive_path: Path, target_dir: Path, split: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        _safe_extract_archive(archive, target_dir, split)


def _prepare_target_dir(target_dir: Path, *, overwrite: bool) -> str | None:
    if not target_dir.exists():
        return "new"
    if not overwrite:
        return None
    shutil.rmtree(target_dir)
    return "replaced"


def stage_split(source: SplitSource, raw_dir: str | Path, *, overwrite: bool = False) -> str:
    """Copy or extract a discovered split into the standard raw layout."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    target_dir = raw_path / source.split

    target_state = _prepare_target_dir(target_dir, overwrite=overwrite)
    if target_state is None:
        return "skipped_existing"

    if source.kind == "directory":
        _copy_split_directory(source.path, target_dir)
        return "replaced" if target_state == "replaced" else "copied"

    if source.kind == "zip":
        _extract_split_archive(source.path, target_dir, source.split)
        return "replaced" if target_state == "replaced" else "extracted"

    raise ValueError(f"Unsupported source kind: {source.kind}")


def build_split_status(
    split: str,
    *,
    raw_dir: Path,
    source: SplitSource | None,
    action: str | None,
) -> dict[str, object]:
    """Build a serializable status record for one supported split."""
    target_dir = raw_dir / split
    present = target_dir.exists()
    resolved_action = action
    if resolved_action is None:
        resolved_action = "existing" if present else "missing"

    return {
        "present": present,
        "source": str(source.path.resolve()) if source is not None else None,
        "source_kind": source.kind if source is not None else None,
        "target_dir": str(target_dir.resolve()),
        "gnt_files": count_gnt_files(target_dir),
        "action": resolved_action,
    }


def summarize_raw_data(
    raw_dir: str | Path,
    *,
    sources: Mapping[str, SplitSource] | None = None,
    actions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Summarize the standard raw layout for every supported split."""
    raw_path = Path(raw_dir)
    sources = dict(sources or {})
    actions = dict(actions or {})

    return {
        "raw_dir": str(raw_path.resolve()),
        "splits": {
            split: build_split_status(
                split,
                raw_dir=raw_path,
                source=sources.get(split),
                action=actions.get(split),
            )
            for split in SUPPORTED_SPLITS
        },
    }


def prepare_raw_data(
    scan_dir: str | Path,
    raw_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Scan for local split sources, stage them into `raw_dir`, and summarize the result."""
    sources = discover_split_sources(scan_dir)
    actions: dict[str, str] = {}

    for split in SUPPORTED_SPLITS:
        source = sources.get(split)
        if source is None:
            continue
        actions[split] = stage_split(source, raw_dir, overwrite=overwrite)

    return summarize_raw_data(raw_dir, sources=sources, actions=actions)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare local CASIA/HWDB zip files or extracted split directories.",
    )
    parser.add_argument(
        "--scan_dir",
        type=Path,
        required=True,
        help="Directory containing CASIA/HWDB zip files and/or extracted split directories.",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory for the normalized raw split layout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing split in raw_dir instead of skipping it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    summary = prepare_raw_data(args.scan_dir, args.raw_dir, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
