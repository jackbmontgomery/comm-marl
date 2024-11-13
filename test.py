import gym
import ma_gym

env_1 = gym.make("TrafficJunction4-v0")
env_2 = gym.make(
    "PredatorPrey7x7-v0",
    grid_shape=(7, 7),
    n_agents=4,
    n_preys=2,
    penalty=-0.75,
)
print()
