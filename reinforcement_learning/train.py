import logging
import datetime as dt

import ray
from ray import tune
from reinforcement_learning.redistribution_env import SimulationEnvironment
import ray.rllib.agents.impala as impala

stop_iters = 30000
stop_reward = 48
stop_timesteps = 10000000000

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    ray.init(dashboard_host="0.0.0.0")

    stop = {
        "training_iteration": stop_iters,
        "timesteps_total": stop_timesteps,
        "episode_reward_mean": stop_reward,
    }
    env_config = {"start_date": dt.datetime(2018, 7, 1), "end_date": dt.datetime(2020, 1, 1),
                  "train": True, 'grep_type': '7_crams', "from_pickle": True, 'use_historic_stocks': True}
    env = SimulationEnvironment(env_config=env_config)

    def policy_mapping_fn(agent_id):
        if agent_id.startswith("low_level_"):
            return "low_level_policy"
        else:
            return "high_level_policy"
    config = impala.DEFAULT_CONFIG.copy()
    print(config)
    config["env"] = SimulationEnvironment
    config["env_config"] = env_config
    config["num_workers"] = 15
    config["num_gpus"] = 0
    #config['model']["fcnet_hiddens"] = [32, 32],

    config["rollout_fragment_length"] = 48
    config["train_batch_size"] = config["rollout_fragment_length"] * 15

    #config["eager_tracing"] = True
    #config["normalize_actions"] = False
    #config["vtrace"] = False
    #config["log_level"] = "WARNING"
    #config["framework"] = "tfe"
    #config["clip_actions"] = True

    results = tune.run("IMPALA", stop=stop, config=config, checkpoint_at_end=True, checkpoint_freq=100,
                       local_dir="/home/ubuntu/teco/02. Python/reinforcement_learning/models")

    ray.shutdown()
