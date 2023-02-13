import gym
import sb3_contrib as sb3c
import stable_baselines3 as sb3
import highway_env

from config import config


def vis():
    env = gym.make('roundabout-v0')
    env.configure(config.time_to_collision)
    model = sb3c.QRDQN.load('checkpoints/qrdqn/time_to_collision')
    for _ in range(10):
        obs = env.reset()
        while True:
            action, state_ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            env.render()
            if done:
                break


if __name__ == '__main__':
    vis()
