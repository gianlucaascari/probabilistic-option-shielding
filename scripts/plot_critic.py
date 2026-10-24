"""Heatmaps of a critic's option safety ``P(safe | s, o)`` over the map.

    python scripts/plot_critic.py --map large_map --critic neural
    python scripts/plot_critic.py --map large_map --critic neural --school-time on --options right,down

One panel per option and school-time value; cells the car cannot drive on are
blank. Defaults to the pretrained critics in ``checkpoints/<map>/``.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch as th  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from logicity.env import load_env_config, make_env  # noqa: E402
from pos.critics import NeuralSafetyCritic, TabularSafetyCritic, load_neural_critic_meta  # noqa: E402

OPTIONS = {"left": 0, "right": 1, "up": 2, "down": 3}
TITLES = ["Option Left ←", "Option Right →", "Option Up ↑", "Option Down ↓"]


def safety_maps(critic_kind, checkpoint, drivable):
    """(2, num_options, H, W) array of P(safe | s, o), NaN on non-drivable cells."""
    h, w = drivable.shape
    rows, cols = np.nonzero(drivable)
    maps = np.full((2, 4, h, w), np.nan)
    for school_time in (0, 1):
        obs = np.stack([rows, cols, np.full_like(rows, school_time)], axis=1).astype(np.float32)
        if critic_kind == "tabular":
            values = TabularSafetyCritic.load(checkpoint)(obs).numpy()
        else:
            meta = load_neural_critic_meta(checkpoint)
            critic = NeuralSafetyCritic(meta["obs_dim"], meta["num_options"], meta["hidden_sizes"],
                                        meta["output_power"])
            critic.load_state_dict(th.load(checkpoint, map_location="cpu"))
            rms = meta["obs_rms"]
            obs = np.clip((obs - rms["mean"]) / np.sqrt(rms["var"] + 1e-8), -10, 10).astype(np.float32)
            with th.no_grad():
                values = critic(th.as_tensor(obs)).numpy()
        maps[school_time][:, rows, cols] = values.T
    return maps


def plot(maps, school_times, options, output, cell_inches):
    finite = np.isfinite(maps).any(axis=(0, 1))
    r = np.flatnonzero(finite.any(axis=1))
    c = np.flatnonzero(finite.any(axis=0))
    maps = maps[:, :, r[0]:r[-1] + 1, c[0]:c[-1] + 1]
    h, w = maps.shape[-2:]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(len(school_times), len(options), squeeze=False,
                             figsize=(len(options) * w * cell_inches + 1.2, len(school_times) * (h * cell_inches + 0.4)))
    for i, school_time in enumerate(school_times):
        for j, option in enumerate(options):
            ax = axes[i, j]
            image = ax.imshow(maps[school_time, option], cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(TITLES[option], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        if len(school_times) > 1:
            axes[i, 0].set_ylabel(f"School time: {'on' if school_time else 'off'}", fontsize=9)
    bar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1])
    bar.set_label("P(safe|s,o)")
    bar.outline.set_visible(False)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", required=True, choices=["large_map", "stochastic_map"])
    parser.add_argument("--critic", required=True, choices=["neural", "tabular"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--school-time", default="both", choices=["both", "on", "off"])
    parser.add_argument("--options", default="left,right,up,down")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint or os.path.join(
        REPO, "checkpoints", args.map, "neural_critic.pt" if args.critic == "neural" else "tabular_critic.npy")
    env = make_env(load_env_config(os.path.join(REPO, "configs", args.map, "env.yaml")))
    drivable = np.asarray(env.agent.movable_region, dtype=bool)
    maps = safety_maps(args.critic, checkpoint, drivable)

    school_times = {"both": [0, 1], "off": [0], "on": [1]}[args.school_time]
    options = [OPTIONS[o.strip()] for o in args.options.split(",")]
    output = args.out or os.path.join("figures", f"{args.map}_{args.critic}_critic.png")
    plot(maps, school_times, options, output, cell_inches=0.06 if args.map == "large_map" else 0.11)


if __name__ == "__main__":
    main()
