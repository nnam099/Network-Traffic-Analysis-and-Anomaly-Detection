from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=True)


def main() -> int:
    try:
        run([sys.executable, "-m", "ruff", "check", "."])
        run([sys.executable, "-m", "compileall", "src", "dashboard", "scripts", "tests"])
        run([sys.executable, "-m", "unittest", "discover", "-f", "-s", "tests"])
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
