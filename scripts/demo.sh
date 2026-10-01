#!/usr/bin/env bash
set -euo pipefail

python train.py --demo --save_dir checkpoints/ --plot_dir plots/ "$@"
