#!/usr/bin/env bash
# Remove local caches and regenerated build output without touching datasets,
# checkpoints or frozen protocol evidence under results/.
set -euo pipefail

cd "$(dirname "$0")/.."

find src tests scripts dashboard -type d -name __pycache__ -prune -exec rm -rf {} +
find src tests scripts dashboard -type f -name '*.py[cod]' -delete
rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov
rm -rf frontend/dist frontend/.vite

echo "Removed Python caches and frontend build output."
echo "Kept: data/, checkpoints/, results/, plots/."
