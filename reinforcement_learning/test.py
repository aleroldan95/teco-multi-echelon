from reinforcement_learning.redistribution_env import SimulationEnvironment
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import ray.rllib.agents.impala as impala
from tqdm import tqdm
import ray
import random
from math import ceil


class TestRun:
    heuristics = ["Action1", "Action2", "Action3", "Action4",
                  'Random', 'No movements', 'Action7', 'Action10', 'Action11', 'Action12',
                  'Action13', 'Neural Net']

    def __init__(self, chosen_heuristics=[], env_config=None):
        self.env = SimulationEnvironment(env_config=env_config)
        self.rewards_low = None

        self.clusters = [1, 2, 3, 4]

        if chosen_heuristics:
            self.chosen_heuristics = chosen_heuristics
        else:
            self.chosen_heuristics = self.heuristics

        self.trainer = None
        if 'Neural Net' in self.chosen_heuristics:
            ray.init(local_mode=True, include_dashboard=False)
            config = impala.impala.DEFAULT_CONFIG.copy()
            config["env"] = SimulationEnvironment
            if env_config:
                config["env_config"] = env_config
            config["num_workers"] = 1
            config["sample_async"] = False
            config["num_gpus"] = 0
            # config['model']["fcnet_hiddens"] = [32, 32],
            config["rollout_fragment_length"] = 48
            config["train_batch_size"] = config["rollout_fragment_length"] * 5
            self.trainer = impala.impala.ImpalaTrainer(config)
            self.trainer.restore(
                "models/IMPALA/IMPALA_SimulationEnvironment_0_2020-09-28_16-23-09va3odfxd/checkpoint_550/checkpoint-550")

    def run(self, slice=10, geus_ids=None):
        self.rewards_low = np.array(
            [[0] * ceil((self.env.end_date - self.env.start_date).days / self.env.days_between_movements)] * len(
                self.chosen_heuristics),
            dtype='float64')
        if geus_ids:
            geus = list(filter(lambda x: x.id in geus_ids,
                               self.env.data_class.geus_by_id.values()))
            random.shuffle(geus)
        elif slice > 0:
            geus = list(filter(lambda x: x.cluster in self.clusters,
                               self.env.data_class.geus_by_id.values()))
            random.shuffle(geus)
            geus = geus[:slice]
        else:
            geus = list(filter(lambda x: x.cluster in self.clusters,
                               self.env.data_class.geus_by_id.values()))
            random.shuffle(geus)
        for geu in tqdm(geus):
            for heuristic in self.chosen_heuristics:
                self._simulate(geu, heuristic)
        self.rewards_low /= slice
        if "Neural Net" in self.chosen_heuristics:
            ray.shutdown()

    def _simulate(self, geu, heuristic):
        self.env.geu = geu
        obs = self.env.reset()

        rewards_low = []
        while True:
            if heuristic == 'Action1':
                action = 0
            elif heuristic == 'Action2':
                action = 1
            elif heuristic == 'Action3':
                action = 2
            elif heuristic == 'Action4':
                action = 3
            elif heuristic == 'No movements':
                action = 4
            elif heuristic == 'Random':
                action = 5
            elif heuristic == 'Action7':
                action = 6
            elif heuristic == 'Action10':
                action = 7
            elif heuristic == 'Action11':
                action = 8
            elif heuristic == 'Action12':
                action = 9
            elif heuristic == 'Action13':
                action = 10
            elif heuristic == 'Neural Net':
                action = self.trainer.compute_action(obs)
            else:
                print(f"Heuristica mal escrita {heuristic}")

            # print(f"Action: {action}")
            obs, reward, done, _ = self.env.step(action)
            # self.env.render()
            # print(f"Observation: {obs}")
            # print(f":Reward {reward}")

            rewards_low.append(reward)

            if done:
                break

        self.rewards_low[self.chosen_heuristics.index(heuristic)] += np.array(rewards_low, dtype="float64")

    def plot_results(self):
        plt.plot(self.rewards_low.T)
        plt.legend(self.chosen_heuristics)
        plt.show()

    def save_results(self):
        plt.plot(self.rewards_low.T)
        plt.legend(self.chosen_heuristics)
        plt.savefig("resultados.png")


if __name__ == "__main__":
    env_config = {"start_date": dt.datetime(2019, 1, 1), "end_date": dt.datetime(2020, 6, 15),
                  'grep_type': '7_crams', "from_pickle": True, 'use_historic_stocks': True}
    test = TestRun(env_config=env_config)
    test.run(slice=40)
    test.plot_results()
    test.save_results()
