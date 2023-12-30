from typing import TextIO
from argparse import ArgumentParser
import json
import os

class Config(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def from_json(cls, path):
        
        with open(path, 'r') as f:
            config = json.load(f)
        
        return cls(config)
    

class Logger:
    
    def __init__(self, base_dir: str, rewards: str, ma_rewards: str, epsilons: str, fixed_states_q: str) -> None:
        
        self.base_dir = base_dir
        self.rewards = rewards
        self.ma_rewards = ma_rewards
        self.epsilons = epsilons
        self.fixed_states_q = fixed_states_q

    def log_rewards(self, rewards):
        
        with open(
            os.path.join(self.base_dir, self.rewards), 'w') as f:
            self.__log_list(f, rewards)

    def log_ma_rewards(self, ma_rewards):
        
        with open(
            os.path.join(self.base_dir, self.ma_rewards), 'w') as f:
            self.__log_list(f, ma_rewards)
    
    
    def log_epsilons(self, epsilons):
        
        with open(
            os.path.join(self.base_dir, self.epsilons), 'w') as f:
            self.__log_list(f, epsilons)

    def log_fixed_states_q(self, fixed_states_q):
        
        with open(
            os.path.join(self.base_dir, self.fixed_states_q), 'w') as f:
            self.__log_list(f, fixed_states_q)

    def __log_list(self, file: TextIO, tgt_list: list) -> None:

        file.write("\n".join([str(r) for r in tgt_list]))



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Path to config file', type=str)
    return parser.parse_args()