from stable_baselines3 import PPO

from .option_ppo import OptionPPO
from .option_ppo_lag import OptionPPOLagrangian
from .pos_ppo import POSPPO
from .ppo_pls import PPOPLS

ALGORITHMS = {
    "PPO": PPO,
    "PPO-PLS": PPOPLS,
    "OptionPPO": OptionPPO,
    "OptionPPO-Lag": OptionPPOLagrangian,
    "POS-PPO": POSPPO,
}
