from stable_baselines3 import PPO
import gym

from snakeEnv import SnekEnv


model_number = 10000
model_dir = f"models/distance/test_{model_number}.zip"

env = SnekEnv()
env.reset()

model = PPO.load(model_dir, env=env, verbose=1)

episodes = 10

for episode in range(episodes):
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action=action)

        print(reward)

