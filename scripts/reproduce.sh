#!/usr/bin/env bash
# Train all methods of one map, then draw its figures. Called by
# reproduce_large_map.sh and reproduce_stochastic_map.sh, which document the options.
set -euo pipefail

MAP=$1; shift
SEEDS=10
FIRST_SEED=10   # seeds 10, 11, ...
JOBS=4
STEPS=""
LOGDIR=logs
PLOT_ONLY=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --seeds) SEEDS=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    --plot-only) PLOT_ONLY=1; shift ;;
    --smoke) SEEDS=1; STEPS=5000; LOGDIR=logs_smoke; shift ;;
    *) echo "Unknown option $1" >&2; exit 1 ;;
  esac
done

PYTHON=${PYTHON:-python}
METHODS="ppo ppo_pls option_ppo option_ppo_rs option_ppo_lag
         pos_ppo_online_neural pos_ppo_pretrained_neural pos_ppo_online_tabular pos_ppo_pretrained_tabular"

if [[ $PLOT_ONLY == 1 ]]; then
  LOGDIR=results/paper_logs
else
  # One process per (method, seed), JOBS at a time; finished runs are skipped.
  run_one() {
    local run=$LOGDIR/$MAP/$1/seed_$2
    [[ -f $run/policy.pt ]] && return 0
    mkdir -p "$run"
    if $PYTHON scripts/train.py --config "configs/$MAP/$1.yaml" --seed "$2" --logdir "$run" \
         ${STEPS:+--total-timesteps $STEPS} > "$run/train.log" 2>&1; then
      echo "done:   $run"
    else
      echo "FAILED: $run (see $run/train.log)"
    fi
  }
  export -f run_one
  export MAP LOGDIR STEPS PYTHON OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  for method in $METHODS; do
    for ((seed = FIRST_SEED; seed < FIRST_SEED + SEEDS; seed++)); do echo "$method $seed"; done
  done | xargs -P "$JOBS" -n 2 bash -c 'run_one "$0" "$1"'
fi

$PYTHON scripts/plot_results.py --map "$MAP" --logdir "$LOGDIR" --out "figures/$MAP"
