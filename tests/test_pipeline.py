"""Sanity checks: `python tests/test_pipeline.py` (runs the offline self-check)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import selfcheck  # noqa: E402

if __name__ == "__main__":
    sys.exit(selfcheck.main())
