from stable_baselines3 import PPO
import os
import gym

from snakeEnv import SnekEnv


models_dir = "models/distance"
logdir = "logs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)


env = SnekEnv()
env.reset()


model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)



for i in range(1, 100):
    model.learn(10000, tb_log_name="distance", reset_num_timesteps=False)
    model.save(f"{models_dir}/test_{10000 * i}")
