import logging
from gym import spaces
from reinforcement_learning.redistribution_env import SimulationEnvironment
from data_loader_class import DataClass

import datetime as dt
import numpy as np

from ray.rllib.env import MultiAgentEnv

logger = logging.getLogger(__name__)


class OrderEnvironment(MultiAgentEnv):

    def __init__(self, env_config={}):
        self.high_level_steps = env_config.get("high_level_steps", 4)
        self.steps_per_high_level = env_config.get("high_level_steps", 12)

        # Creo low_level_agent
        self.distribution_env = SimulationEnvironment(env_config)

        # Environment Spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float16)
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=1, shape=(self.distribution_env.num_wh, 3)),
                                               spaces.Discrete(4)))

        # Instance Variables
        self.high_level_step_num = 0
        self.low_level_step_num = 0
        self.temp_reward = 0

    def reset(self):
        self.high_level_step_num = 0
        self.low_level_step_num = 0
        self.temp_reward = 0
        return {"high_level_agent": self.distribution_env.reset()}

    def soft_reset(self):
        self.high_level_step_num = 0
        self.low_level_step_num = 0
        self.temp_reward = 0
        return {"high_level_agent": self.distribution_env.soft_reset()}

    def step(self, action_dict):
        assert len(action_dict) == 1, action_dict
        if "high_level_agent" in action_dict:
            return self._high_level_step(action_dict["high_level_agent"])
        else:
            return self._low_level_step(list(action_dict.values())[0])

    def render(self, mode="human"):
        self.distribution_env.render(mode)

    def _high_level_step(self, action):
        # logger.debug("High level agent purchases".format(action))
        self.high_level_step_num += 1
        self.low_level_step_num = 0
        self.temp_reward = 0
        self.low_level_agent_id = "low_level_{}".format(self.high_level_step_num)

        self.distribution_env.purchase_items(action)

        obs = {self.low_level_agent_id: self.distribution_env.get_obs()}
        rew = {self.low_level_agent_id: 0}
        done = {"__all__": False}
        return obs, rew, done, {}

    def _low_level_step(self, action):
        logger.debug("Low level agent step {}".format(action))
        self.low_level_step_num += 1

        # Step in the actual env
        low_level_obs, low_level_rew, low_level_done, low_level_info = self.distribution_env.step(action)

        # Assign low-level agent observation and reward
        obs = {self.low_level_agent_id: low_level_obs}
        rew = {self.low_level_agent_id: low_level_rew}

        # Add low_level_rew to high level
        self.temp_reward += low_level_rew

        # Handle env termination & transitions back to higher level
        done = {"__all__": False}
        if low_level_done:
            done["__all__"] = True
            logger.debug("high level final reward {}".format(self.temp_reward))
            rew["high_level_agent"] = self.temp_reward
            obs["high_level_agent"] = low_level_obs
        elif self.low_level_step_num == self.steps_per_high_level:
            done[self.low_level_agent_id] = True
            rew["high_level_agent"] = self.temp_reward - sum(self.distribution_env.stock_by_wh) / self.distribution_env.max_order
            obs["high_level_agent"] = low_level_obs

        return obs, rew, done, low_level_info
