"""Train one method on one map with one seed.

    python scripts/train.py --config configs/large_map/pos_ppo_pretrained_neural.yaml --seed 0

TensorBoard and CSV logs go to ``--logdir`` (default
``logs/<map>/<method>/seed_<seed>``), together with the final policy, the
VecNormalize statistics and, for online POS-PPO, the learned critic.
"""
import argparse
import os
import sys

import yaml
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logicity.env import load_env_config, make_env  # noqa: E402
from pos import ALGORITHMS, Shield  # noqa: E402
from pos.callbacks import SafetyMetricsCallback  # noqa: E402
from pos.policies import MLPFeatureExtractor  # noqa: E402


def resolve(path, base):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def build_vec_env(env_config, seed, device):
    def init():
        env = make_env(env_config, device=device)
        env.seed(seed)
        env.init()
        return env

    vec_env = VecNormalize(DummyVecEnv([init]), norm_obs=True, norm_reward=True, clip_obs=10.0)
    vec_env.reset()
    return vec_env


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--total-timesteps", type=int, default=None, help="override the config")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_random_seed(args.seed)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    map_name = os.path.basename(config_dir)
    method = os.path.splitext(os.path.basename(args.config))[0]
    logdir = args.logdir or os.path.join("logs", map_name, method, f"seed_{args.seed}")

    env_config = load_env_config(resolve(config["env"], config_dir), config.get("env_overrides"))
    env = build_vec_env(env_config, args.seed, args.device)

    hyperparameters = dict(config.get("hyperparameters") or {})
    if "critic_path" in hyperparameters:
        hyperparameters["critic_path"] = resolve(hyperparameters["critic_path"], config_dir)
    if "shield" in config:
        hyperparameters["shield"] = Shield(resolve(config["shield"], config_dir),
                                           get_sensor_values=env.venv.envs[0].get_sensor_values,
                                           device=args.device)
    policy_kwargs = {"features_extractor_class": MLPFeatureExtractor,
                     "features_extractor_kwargs": {"features_dim": config.get("features_dim", 128)}}

    algorithm = ALGORITHMS[config["algorithm"]]
    model = algorithm("MlpPolicy", env, policy_kwargs=policy_kwargs, device=args.device, verbose=1,
                      **hyperparameters)
    model.set_logger(configure(logdir, ["stdout", "csv", "tensorboard"]))
    model.learn(total_timesteps=args.total_timesteps or config["total_timesteps"],
                callback=SafetyMetricsCallback())

    model.policy.save(os.path.join(logdir, "policy.pt"))
    env.save(os.path.join(logdir, "vecnormalize.pkl"))
    if getattr(model, "online", False):
        model.save_critic(os.path.join(logdir, "critic.npy" if model.critic_kind == "tabular" else "critic.pt"))


if __name__ == "__main__":
    main()
