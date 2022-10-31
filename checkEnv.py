from stable_baselines3.common.env_checker import check_env
from snakeEnv import SnekEnv


env = SnekEnv()

check_env(env)