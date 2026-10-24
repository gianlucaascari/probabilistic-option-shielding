#!/usr/bin/env bash
# Reproduce the Large Map results: Figure 3 and Figure 10 (training curves of all
# methods) and Figures 2a, 6, 7 (option safety of the pretrained critics).
#
#   ./reproduce_large_map.sh                 # 9 methods x 10 seeds (10-19), 4 runs in parallel
#   ./reproduce_large_map.sh --jobs 10       # more parallel runs (one CPU core each)
#   ./reproduce_large_map.sh --seeds 3       # fewer seeds
#   ./reproduce_large_map.sh --smoke         # 1 seed x 5000 steps, to check the setup
#   ./reproduce_large_map.sh --plot-only     # redraw the figures from the paper's logs
#
# Training logs go to logs/large_map/, figures to figures/large_map/.
set -euo pipefail
cd "$(dirname "$0")"
bash scripts/reproduce.sh large_map "$@"
PYTHON=${PYTHON:-python}
$PYTHON scripts/plot_critic.py --map large_map --critic neural --school-time on --options right,down \
  --out figures/large_map/figure2a.png
$PYTHON scripts/plot_critic.py --map large_map --critic tabular --out figures/large_map/figure6.png
$PYTHON scripts/plot_critic.py --map large_map --critic neural --out figures/large_map/figure7.png
