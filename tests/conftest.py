from pathlib import Path
import shutil
import sys
import tempfile
from uuid import uuid4

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@pytest.fixture
def tmp_path() -> Path:
    """Writable temp directory using system temp (avoids Chinese-path issues with cv2.imwrite)."""
    path = Path(tempfile.mkdtemp(prefix="hwtest_"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
