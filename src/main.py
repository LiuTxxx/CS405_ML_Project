import os
import sys
from multiprocessing import Pool, set_start_method

import gym
import highway_env
import torch

from config import config
from rl_method import run_train_and_eval, get_algorithm_name


def _eval_one(name, alg):
    env = gym.make('intersection-v0')
    env.configure(config[name])
    env.reset()
    run_train_and_eval(env, name, alg)


def eval_all():
    reg_names = get_algorithm_name()
    cfg_types = [name for name in config]
    for reg in reg_names:
        with Pool(3) as p:
            p.starmap(_eval_one, [(cfg, reg) for cfg in cfg_types])


if __name__ == '__main__':
    if sys.platform == 'linux':
        set_start_method('spawn')
    eval_all()
