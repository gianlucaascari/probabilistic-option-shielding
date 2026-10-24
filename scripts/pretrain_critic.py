"""Pretrain an option-safety critic: collect option executions, then fit.

    python scripts/pretrain_critic.py --config configs/large_map/pretrain_neural.yaml

Collection runs options chosen without any learned policy and labels every
executed action with its PLS safety probability. The dataset is cached at
``--dataset`` and reused if it exists. The fitted critic is written to
``--out`` (``.pt`` + ``.json`` for neural critics, ``.npy`` for tabular ones).
"""
import argparse
import os
import pickle
import sys

import numpy as np
import yaml
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logicity.env import load_env_config, make_env  # noqa: E402
from logicity.options import IntersectionOptions  # noqa: E402
from pos import Shield  # noqa: E402
from pos.critics import save_neural_critic  # noqa: E402
from pos.pretraining import fit_neural_critic, fit_tabular_critic  # noqa: E402

class OptionSampler:
    """Executes options chosen uniformly at random, or with count-based
    exploration (``count_ucb``: option ``o`` in state ``s`` is drawn with
    probability proportional to ``(N(s, o) + 1) ** -ucb_beta``)."""

    def __init__(self, vec_env, shield, num_options=4, exploration="uniform", ucb_beta=1.0):
        self.env = vec_env
        self.shield = shield
        self.num_options = num_options
        self.exploration = exploration
        self.ucb_beta = ucb_beta
        self.counts = {}
        self.options = IntersectionOptions.from_env(vec_env)

    def _raw(self, obs):
        return self.env.unnormalize_obs(obs.copy())[0]

    def _choose_option(self, obs):
        if self.exploration == "uniform":
            return np.random.randint(self.num_options)
        state = tuple(int(round(x)) for x in self._raw(obs)[-3:])
        counts = self.counts.setdefault(state, np.zeros(self.num_options, dtype=np.int64))
        weights = 1.0 / np.power(counts + 1.0, self.ucb_beta)
        option = int(np.random.choice(self.num_options, p=weights / weights.sum()))
        counts[option] += 1
        return option

    def collect(self, steps):
        """Return the transitions of every option execution that was not cut
        short by the episode step limit."""
        transitions, n_steps = [], 0
        obs = self.env.reset()
        progress = tqdm(total=steps, desc="Collecting", unit="step")
        while n_steps < steps:
            option = self._choose_option(obs)
            self.options.start(self._raw(obs), option)
            execution = []
            while n_steps < steps:
                action = self.options.action(self._raw(obs), option)
                sensors = self.shield.get_sensor_values()
                raw_obs = self.env.get_original_obs()[0].copy()
                new_obs, _, dones, infos = self.env.step(np.array([action]))
                n_steps += 1
                progress.update(1)
                # Under inertia the executed action may differ from the chosen one.
                executed = int(infos[0].get("inertia_executed_action", action))
                terminated = self.options.terminated(self._raw(new_obs), option)
                execution.append({
                    "obs": obs[0].copy(),
                    "next_obs": new_obs[0].copy(),
                    "unnorm_obs": raw_obs,
                    "unnorm_next_obs": self.env.get_original_obs()[0].copy(),
                    "option": option,
                    "safe_reward": float(self.shield.action_safeties(sensors)[0, executed]),
                    "next_beta": 1.0 if terminated else 0.0,
                    "done": dones[0],
                })
                obs = new_obs
                if dones[0] or terminated:
                    if not infos[0].get("overtime", False):
                        transitions.extend(execution)
                    break
        progress.close()
        return transitions


def collect_dataset(config, config_dir, seed):
    set_random_seed(seed)
    env_config = load_env_config(os.path.join(config_dir, config["env"]))
    collect = config["collect"]

    def init():
        env = make_env(env_config, random_start=collect.get("random_start", False))
        env.seed(seed)
        env.init()
        return env

    vec_env = VecNormalize(DummyVecEnv([init]), norm_obs=True, norm_reward=True, clip_obs=10.0)
    shield = Shield(os.path.join(config_dir, config["shield"]),
                    get_sensor_values=vec_env.venv.envs[0].get_sensor_values)
    sampler = OptionSampler(vec_env, shield, exploration=collect.get("exploration", "uniform"),
                            ucb_beta=collect.get("ucb_beta", 1.0))
    transitions = sampler.collect(collect["steps"])
    return {"transitions": transitions,
            "obs_rms": {"mean": vec_env.obs_rms.mean.copy(), "var": vec_env.obs_rms.var.copy()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None, help="dataset cache (default: the config's `dataset`)")
    parser.add_argument("--out", default=None, help="output critic (default: checkpoints/<map>/pretrained_<critic>)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the data collection")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    map_name = os.path.basename(config_dir)
    dataset_path = args.dataset or config["dataset"]
    fit = config["fit"]

    if os.path.exists(dataset_path):
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
        print(f"Loaded {len(dataset['transitions'])} transitions from {dataset_path}")
    else:
        dataset = collect_dataset(config, config_dir, args.seed)
        os.makedirs(os.path.dirname(dataset_path) or ".", exist_ok=True)
        with open(dataset_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Saved {len(dataset['transitions'])} transitions to {dataset_path}")

    set_random_seed(fit.get("seed", 0))
    if fit["critic"] == "neural":
        out = args.out or os.path.join("checkpoints", map_name, "pretrained_neural_critic.pt")
        model = fit_neural_critic(
            dataset["transitions"], num_options=4, hidden_sizes=fit["hidden_sizes"],
            output_power=fit["output_power"], lr=fit["lr"], gradient_steps=fit["gradient_steps"],
            batch_size=fit.get("batch_size", 128), polyak_tau=fit.get("polyak_tau", 0.005),
            lr_plateau=fit.get("lr_plateau"), sampling_seed=fit.get("sampling_seed", 42))
        save_neural_critic(model, out, obs_rms=dataset["obs_rms"], hidden_sizes=fit["hidden_sizes"])
    else:
        out = args.out or os.path.join("checkpoints", map_name, "pretrained_tabular_critic.npy")
        critic = fit_tabular_critic(
            dataset["transitions"], fit["table_shape"], num_options=4, alpha=fit.get("alpha", 0.1),
            max_passes=fit.get("max_passes", 10_000), tolerance=fit.get("tolerance", 1e-4),
            plateau=fit.get("plateau"))
        critic.save(out)
    print(f"Critic written to {out}")


if __name__ == "__main__":
    main()
