from pathlib import Path
import shutil
import sys
from uuid import uuid4

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TEST_TEMP_ROOT = PROJECT_ROOT / "tests_tmp" / "pytest-workspace"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@pytest.fixture
def tmp_path() -> Path:
    """Project-scoped writable temp directory for Windows environments with flaky system temp permissions."""
    TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_TEMP_ROOT / f"case_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
