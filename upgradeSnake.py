from stable_baselines3 import PPO
import gym

from snakeEnv import SnekEnv


model_number = 10000
model_dir = f"models/distance/test_{model_number}.zip"

models_dir = "models/upgraded"
logdir = "logs"

env = SnekEnv()


model = PPO.load(model_dir, env=env, verbose=1)

TIMESTEPS = 10000

for i in range(int(model_number / TIMESTEPS), 10):
    model.learn(TIMESTEPS, tb_log_name="upgraded", reset_num_timesteps=False)
    model.save(f"{models_dir}/test_{TIMESTEPS * i}")


