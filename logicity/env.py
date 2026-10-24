"""Build a single-agent LogiCity environment from an environment YAML."""
import os

import yaml

from .utils.gym_wrapper import GymCityWrapper
from .utils.inertia_wrapper import InertiaWrapper
from .utils.load import CityLoader
from .utils.random_start_wrapper import RandomStartWrapper

_PATH_KEYS = ("map_yaml_file", "agent_yaml_file", "ontology_yaml_file", "rule_yaml_file")


def load_env_config(path, overrides=None):
    """Read an environment YAML, apply ``overrides`` and resolve its file
    paths relative to the YAML's directory."""
    with open(path) as f:
        config = yaml.safe_load(f)
    config.update(overrides or {})
    base = os.path.dirname(os.path.abspath(path))
    for key in _PATH_KEYS:
        config[key] = os.path.join(base, config[key])
    return config


def make_env(config, device="cpu", random_start=False):
    """GymCityWrapper, optionally with random option-endpoint starts and
    with sticky-action inertia when ``config["inertia"]["prob"] > 0``."""
    city, _ = CityLoader.from_yaml(
        **{k: v for k, v in config.items() if k != "inertia"}, device=device)
    env = GymCityWrapper(city, device=device)
    if random_start:
        env = RandomStartWrapper(env)
    inertia = config.get("inertia") or {}
    if inertia.get("prob", 0.0) > 0:
        env = InertiaWrapper(env, prob=inertia["prob"])
    return env
