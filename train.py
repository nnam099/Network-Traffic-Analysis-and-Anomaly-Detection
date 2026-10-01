"""Train or demo the IDS model from the repository root."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ids.training import main, run_demo, run_full, save_artifacts  # noqa: E402,F401

if __name__ == "__main__":
    main()
