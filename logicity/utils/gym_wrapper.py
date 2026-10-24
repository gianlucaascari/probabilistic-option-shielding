import logging

import gym
import numpy as np
import torch

from ..core.config import *

logger = logging.getLogger(__name__)


class GymCityWrapper(gym.core.Env):
    """Single-agent Gym interface over a CityEnv.

    The observation is ``[row, col, school_time]`` of the controlled agent.
    Besides the usual ``(obs, reward, done, info)``, every step reports the
    task rules violated in that step (``info["Violations"]``) and a
    constrained-MDP cost (``info["cost"]``, one per violated ``dead: true``
    rule). When an episode ends, ``info["episode"]`` carries its return,
    discounted return, length, violation counts and total cost; these are the
    quantities reported in the paper.
    """

    def __init__(self, env, device="cpu"):
        self.device = device
        self.env = env
        self.logic_grounding_shape = self.env.logic_grounding_shape
        self.pred_grounding_index = self.env.pred_grounding_index
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.logic_grounding_shape,), dtype=np.float32)

        rl_agent = env.rl_agent
        self.agent_name = rl_agent["agent_name"]
        self.horizon = rl_agent["max_horizon"]
        self.agent_type, agent_id = self.agent_name.split("_")
        self.agent_layer_id = None
        for agent in self.env.agents:
            if agent.type == self.agent_type and agent.id == int(agent_id):
                self.agent = agent
                self.agent_layer_id = agent.layer_id
        assert self.agent_layer_id is not None, "Agent not found! Recheck agent_name in the config file."

        self.action_space = gym.spaces.Discrete(rl_agent["action_space"])
        self.action_mapping = rl_agent["action_mapping"]
        self.max_priority = rl_agent["max_priority"]
        self.action_cost = rl_agent["action_cost"]
        self.reset_dist = rl_agent.get("reset_dist")
        self.overtime_cost = rl_agent.get("overtime_cost", -3)
        self.type2label = {v: k for k, v in LABEL_MAP.items()}
        self.t = 0

        # Discount used only to report the episodic discounted return.
        self.gamma = rl_agent.get("gamma", 0.99)
        # When reward shaping is active, report the episode return without the
        # shaping penalties so that all methods are scored on the task reward.
        self.exclude_safety_penalty_from_ep_rew_mean = rl_agent.get(
            "exclude_safety_penalty_from_ep_rew_mean", False)
        task_rules = getattr(self.env.local_planner, "rules", {}).get("Task", {})
        self._task_rules = {k: v for k, v in task_rules.items() if isinstance(v, dict)}
        # Violating a `dead: true` rule ends the episode; each such violation
        # costs 1 in the constrained-MDP formulation.
        self.cost_rule_names = {k for k, v in self._task_rules.items() if v.get("dead")}
        self.safety_penalty_rule_names = {
            k for k in self.cost_rule_names if float(self._task_rules[k].get("reward", 0.0)) < 0}
        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self.current_episode_reward = 0
        self.current_episode_logged_reward = 0
        self.current_episode_discounted_reward = 0
        self.current_episode_length = 0
        self.current_episode_violations = {}
        self.current_episode_cost = 0.0

    def _sum_safety_penalties(self, violations):
        return sum(float(self._task_rules[v].get("reward", 0.0))
                   for v in violations if v in self.safety_penalty_rule_names)

    def _episode_info(self):
        return {
            'r': self.current_episode_logged_reward,
            'r_disc': self.current_episode_discounted_reward,
            'l': self.current_episode_length,
            'violations': self.current_episode_violations,
            'c': self.current_episode_cost,
        }

    def _flatten_obs(self, obs_dict):
        return obs_dict["World_state"][0].astype(np.float32)

    def _get_reward(self, obs_dict, action):
        if obs_dict["Fail"][0]:
            return obs_dict["Reward"][0]
        return self.action_cost[action] + obs_dict["Reward"][0]

    def reset(self):
        self.t = 0
        self.agent.init(self.env.city_grid)
        self.agent.reset_concepts(self.max_priority, self.reset_dist)
        agent_code = self.type2label[self.agent_type]
        self.env.local_planner.reset()
        self.env.random_pedestrian()
        agent_layer = torch.zeros((self.env.grid_size[0], self.env.grid_size[1]))
        start, goal = self.agent.start, self.agent.goal
        agent_layer[start[0], start[1]] = agent_code
        agent_layer[goal[0], goal[1]] = agent_code + AGENT_GOAL_PLUS
        self.env.city_grid[self.agent_layer_id] = agent_layer
        ob_dict = self.env.update(self.agent_layer_id)
        self.obs_dict = ob_dict
        self.current_obs = self._flatten_obs(ob_dict)
        self._reset_episode_stats()
        return self.current_obs

    def init(self):
        """First reset of a freshly built city (does not re-sample the agent)."""
        self.t = 0
        self.env.store_school_pedestrians()
        self.env.random_pedestrian()
        self.env.local_planner.reset()
        ob_dict = self.env.update(self.agent_layer_id)
        self.current_obs = self._flatten_obs(ob_dict)
        return self.current_obs

    def action_masks(self):
        valid_ll_actions = self.env.get_rl_agent_valid_actions(self.agent_layer_id)
        if not isinstance(valid_ll_actions, torch.Tensor):
            valid_ll_actions = torch.tensor(valid_ll_actions, device=self.device)
        if not hasattr(self, 'action_mapping_tensor'):
            self.action_mapping_tensor = torch.stack([
                torch.tensor(self.action_mapping[k], dtype=torch.int32)
                for k in sorted(self.action_mapping.keys())
            ])
        if self.action_mapping_tensor.device != valid_ll_actions.device:
            self.action_mapping_tensor = self.action_mapping_tensor.to(valid_ll_actions.device)
        # An action is valid if all the low-level moves it requires are valid.
        missing_requirements = self.action_mapping_tensor * (1 - valid_ll_actions.int())
        return missing_requirements.sum(dim=1) == 0

    def step(self, action):
        self.t += 1
        info = {}
        one_hot_action = torch.tensor(self.action_mapping[action], dtype=torch.float32)
        # Non-immediate rules are evaluated on the pre-move state inside
        # move_rl_agent; immediate rules (e.g. collisions) after the move.
        current_obs = self.env.move_rl_agent(one_hot_action, self.agent_layer_id)
        rew = self._get_reward(current_obs, action)
        current_violations = list(current_obs.get("Violations", []))
        info.update(current_obs)

        new_ob_dict = self.env.update(self.agent_layer_id)
        immediate_fail, immediate_rew, immediate_violations = self.env.local_planner.eval(one_hot_action, True)
        rew += immediate_rew
        current_obs.setdefault("Violations", []).extend(immediate_violations)
        for v in current_obs["Violations"]:
            self.current_episode_violations[v] = self.current_episode_violations.get(v, 0) + 1

        step_cost = float(sum(1 for v in current_obs["Violations"] if v in self.cost_rule_names))
        self.current_episode_cost += step_cost
        info["cost"] = step_cost
        info["predicates"] = new_ob_dict['Ground_dic'][0]

        self.current_episode_reward += rew
        step_log_reward = rew
        if self.exclude_safety_penalty_from_ep_rew_mean:
            step_log_reward -= self._sum_safety_penalties(current_violations)
            step_log_reward -= self._sum_safety_penalties(immediate_violations)
        self.current_episode_logged_reward += step_log_reward
        self.current_episode_discounted_reward += (self.gamma ** self.current_episode_length) * step_log_reward
        self.current_episode_length += 1
        self.current_obs = self._flatten_obs(new_ob_dict)

        done = self.agent.reach_goal or current_obs["Fail"][0] or immediate_fail
        info["is_success"] = bool(self.agent.reach_goal)
        if done:
            info['episode'] = self._episode_info()
            self.reset()
        if self.t >= self.horizon:
            done = True
            rew += self.overtime_cost
            info["overtime"] = True
            info['episode'] = self._episode_info()
            self.reset()
        if info["Fail"][0]:
            done = True
            info['episode'] = self._episode_info()
            self.reset()
        return self.current_obs, rew, done, info

    def get_sensor_values(self):
        return self.env.get_sensor_values(self.agent_layer_id)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
