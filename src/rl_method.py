import os

from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import QRDQN


_all_algorithms = {}
_save_dir = './checkpoint_is'
_tb_log_dir = './tb_log_intersect/'


def _register_algorithm(name, function, override=False):
    if name not in _all_algorithms or override:
        _all_algorithms[name] = function
    else:
        raise ValueError(
            f'{name} already registered. Set override=True to override old registration'
        )


def get_algorithm_name():
    return [name for name in _all_algorithms]


def run_train_and_eval(env, name, algorithm):
    """
    the main entrance of the rl method
    :param env: the test env
    :param name: environment name
    :param algorithm: string. Should be registered first use {register_algorithm}
    :return: evaluation result
    """
    assert algorithm in _all_algorithms, f'{algorithm} is not registered. '
    _all_algorithms[algorithm](env, name)


def dqn(env, name):
    model = DQN('MlpPolicy',
                env,
                verbose=0,
                gamma=0.8,
                learning_starts=200,
                target_update_interval=50,
                train_freq=1,
                gradient_steps=1,
                policy_kwargs=dict(net_arch=[400, 300]),
                tensorboard_log=_tb_log_dir)
    model.learn(total_timesteps=20000, tb_log_name=f'{name}_dqn')
    model.save(os.path.join(_save_dir, 'dqn', name))


def ppo(env, name):
    model = PPO("MlpPolicy",
                env,
                verbose=0,
                gamma=0.8,
                policy_kwargs=dict(net_arch=[400, 300]),
                tensorboard_log=_tb_log_dir)
    model.learn(total_timesteps=20000, tb_log_name=f'{name}_ppo')
    model.save(os.path.join(_save_dir, 'ppo', name))


def qrdqn(env, name):
    model = QRDQN("MlpPolicy",
                  env,
                  verbose=0,
                  gamma=0.8,
                  learning_starts=200,
                  target_update_interval=50,
                  train_freq=1,
                  gradient_steps=1,
                  policy_kwargs=dict(net_arch=[400, 300]),
                  tensorboard_log=_tb_log_dir)
    model.learn(total_timesteps=20000, tb_log_name=f'{name}_qrdqn')
    model.save(os.path.join(_save_dir, 'qrdqn', name))


def a2c(env, name):
    model = A2C("MlpPolicy",
                env,
                verbose=0,
                gamma=0.8,
                policy_kwargs=dict(net_arch=[400, 300]),
                tensorboard_log=_tb_log_dir)
    model.learn(total_timesteps=20000, tb_log_name=f'{name}_a2c')
    model.save(os.path.join(_save_dir, 'a2c', name))


_register_algorithm('PPO', ppo)
_register_algorithm('QRDQN', qrdqn)
_register_algorithm('DQN', dqn)
_register_algorithm('A2C', a2c)
