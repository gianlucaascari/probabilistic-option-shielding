#!/usr/bin/env bash
# Reproduce the Stochastic Map results: Figure 4 and Figure 11 (training curves
# of all methods) and Figures 2b, 8, 9 (option safety of the pretrained critics).
#
#   ./reproduce_stochastic_map.sh                 # 9 methods x 10 seeds (10-19), 4 runs in parallel
#   ./reproduce_stochastic_map.sh --jobs 10       # more parallel runs (one CPU core each)
#   ./reproduce_stochastic_map.sh --seeds 3       # fewer seeds
#   ./reproduce_stochastic_map.sh --smoke         # 1 seed x 5000 steps, to check the setup
#   ./reproduce_stochastic_map.sh --plot-only     # redraw the figures from the paper's logs
#
# Training logs go to logs/stochastic_map/, figures to figures/stochastic_map/.
set -euo pipefail
cd "$(dirname "$0")"
bash scripts/reproduce.sh stochastic_map "$@"
PYTHON=${PYTHON:-python}
$PYTHON scripts/plot_critic.py --map stochastic_map --critic neural --school-time on --options left,up \
  --out figures/stochastic_map/figure2b.png
$PYTHON scripts/plot_critic.py --map stochastic_map --critic tabular --out figures/stochastic_map/figure8.png
$PYTHON scripts/plot_critic.py --map stochastic_map --critic neural --out figures/stochastic_map/figure9.png
