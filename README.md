# Safe Reinforcement Learning via Probabilistic Option Shielding

Code for **Probabilistic Option Shielding (POS)**.

Probabilistic Logic Shields (PLS) make a policy safer by reweighting each action
by its probability of being safe, `P(safe | s, a)`, computed by a probabilistic
logic program. This only looks one step ahead. POS lifts the shield to
*options*, temporally extended actions, by learning an **option-safety critic**
`P(safe | s, o)`: the probability that a whole execution of option `o` from `s`
stays safe. The critic is learned by temporal-difference updates from the
one-step PLS signals, so no model of the environment is needed. It is either
**tabular** or **neural**, and either **pretrained** on data from any policy or
learned **online** together with the policy. The option-level policy is then
shielded as in PLS:

```
Π+(o | s) = P(safe | s, o) Π(o | s) / Σ_o' P(safe | s, o') Π(o' | s)
```

and trained with PPO plus the PLS safety loss `-log Σ_o P(safe | s, o) Π(o | s)`.

## Repository layout

| Path | Contents |
|---|---|
| `pos/` | The method, independent of the environment |
| `pos/shield.py` | PLS: ProbLog program → `P(safe \| s, a)` |
| `pos/critics.py` | Option-safety critics (neural, tabular) and the TD target (Eq. 5) |
| `pos/algorithms/pos_ppo.py` | **POS-PPO**: option-level PPO shielded by the critic |
| `pos/algorithms/` | Baselines: `PPOPLS`, `OptionPPO`, `OptionPPOLagrangian` (PPO is Stable-Baselines3's) |
| `pos/buffers.py` | Semi-MDP rollout buffers (returns discounted by `γ^duration`) |
| `pos/pretraining.py` | Offline fitting of the critics |
| `logicity/` | The LogiCity environment used in the paper (see [Attribution](#attribution)) |
| `logicity/options.py` | The four options: drive to the next intersection left, right, up or down |
| `configs/{large_map,stochastic_map}/` | One config per method and map, environment files, safety programs |
| `checkpoints/` | The pretrained critics used in the paper |
| `results/paper_logs/` | Training curves of all paper runs (10 seeds × 9 methods × 2 maps) |
| `scripts/` | `train.py`, `pretrain_critic.py`, `plot_results.py`, `plot_critic.py` |
| `reproduce_large_map.sh`, `reproduce_stochastic_map.sh` | Reproduce the paper's results |

Methods and their configs (`configs/<map>/<name>.yaml`):

| Paper name | Config |
|---|---|
| PPO | `ppo` |
| PPO-PLS | `ppo_pls` |
| OptionPPO | `option_ppo` |
| OptionPPO-RS | `option_ppo_rs` |
| OptionPPO-Lag | `option_ppo_lag` |
| POS-PPO Online / Pretrained (neural critic) | `pos_ppo_online_neural`, `pos_ppo_pretrained_neural` |
| POS-PPO Online / Pretrained (tabular critic) | `pos_ppo_online_tabular`, `pos_ppo_pretrained_tabular` |

## Installation

Python 3.11 is required. The ProbLog circuits are compiled with
[kompyle](https://github.com/ML-KULeuven/amc-grad), which is built from source
and needs a [Rust toolchain](https://rustup.rs/).

```bash
conda create -n pos python=3.11
conda activate pos
# pyastar2d builds against numpy, which must be installed first
pip install numpy==1.26.4 setuptools wheel
pip install --no-build-isolation pyastar2d==1.0.6
pip install -e .
```

## Reproducing the results

Each map has one script. It trains the nine methods for 500k environment
steps with 10 seeds (seeds 10 to 19), then draws the figures into `figures/<map>/`:

```bash
./reproduce_large_map.sh         # Figures 3 and 10, critic heatmaps (Figures 2a, 6, 7)
./reproduce_stochastic_map.sh    # Figures 4 and 11, critic heatmaps (Figures 2b, 8, 9)
```

| Option | Effect |
|---|---|
| `--jobs N` | runs in parallel (default 4); each run uses one CPU core |
| `--seeds N` | seeds per method, starting from seed 10 (default 10) |
| `--smoke` | 1 seed × 5000 steps into `logs_smoke/`, to check the installation |
| `--plot-only` | no training: draw the figures from the paper's logs in `results/paper_logs/` |

One run takes 10 to 20 minutes on a single core of an Apple M1 Pro (Table 6 of
the paper), so a full map is 90 runs. Interrupted reproductions resume: finished
runs are skipped. A single run can also be launched directly:

```bash
python scripts/train.py --config configs/large_map/pos_ppo_online_neural.yaml --seed 0
```

Logs (`progress.csv` and TensorBoard) go to `logs/<map>/<method>/seed_<k>/`. The
plotted metrics are `rollout/ep_disc_rew_mean` (episodic discounted return) and
`rollout/cum_total_violations` (safety violations since the start of training).
Runs are deterministic for a given seed on CPU; they do not match the paper's
logs bit for bit, because those runs also rendered diagnostic plots that drew
from the same random number generators.

### Pretraining the critics

The pretrained methods load the critics in `checkpoints/`. To pretrain them
again, collect the option executions and fit the critic:

```bash
python scripts/pretrain_critic.py --config configs/large_map/pretrain_neural.yaml
```

The critic is written to `checkpoints/<map>/pretrained_<type>_critic.*`; point
`critic_path` in a POS-PPO config to it to use it. Re-pretrained critics match
the shipped ones in behaviour but not bit for bit: the original data collection
was not seeded.

## Citation

```bibtex
@misc{ascari2026pos,
  title        = {Safe Reinforcement Learning via Probabilistic Option Shielding},
  author       = {Ascari, Gian Luca and Tamassia, Isidoro and Jiao, Ying and De Raedt, Luc and Marra, Giuseppe},
  howpublished = {NeurIPS 2026 Workshop on Neuro-Symbolic Embodied Intelligence (NEmo)},
  year         = {2026},
  url          = {https://openreview.net/forum?id=hrECqgwr7h}
}
```

## Attribution

The environment in `logicity/` is derived from
[LogiCity](https://github.com/Jaraxxus-Me/LogiCity) (MIT licence). It keeps the
parts of LogiCity used in the paper and adds the hierarchical car agent, the
military zone, school-time pedestrians, sticky-action inertia, the option
controllers and the safety-violation bookkeeping of the Gym wrapper.

```bibtex
@inproceedings{li2024logicity,
  title     = {{LogiCity}: Advancing Neuro-Symbolic {AI} with Abstract Urban Simulation},
  author    = {Li, Bowen and Li, Zhaoyu and Du, Qiwei and Luo, Jinqi and Wang, Wenshan and Xie, Yaqi and
               Stepputtis, Simon and Wang, Chen and Sycara, Katia P and Ravikumar, Pradeep Kumar and
               Gray, Alex and Si, Xujie and Scherer, Sebastian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}
```

Probabilistic Logic Shields: Yang et al., *Safe Reinforcement Learning via
Probabilistic Logic Shields*, IJCAI 2023.
