import lbforaging
import gymnasium as gym
import numpy as np
import random

class QLearner():

    def __init__(self, num_observations, num_actions, eps_decay=0.995, eps_min=0.1, alpha=1, gamma=0.99):

        self.num_observations = num_observations
        self.num_actions = num_actions

        self.eps = 1
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        self.q_table = np.zeros((num_observations, num_actions))

        self.alpha = alpha
        self.gamma = gamma

    def act(self, obs):
        if random.random() < self.eps:
            action = np.random.randint(0 , self.num_actions)
        else:
            actions_values = self.q_table[obs, :]
            action = np.argmax(actions_values)

        self.eps = max(self.eps * self.eps_decay, self.eps_min)

        return action

    def update(self, obs, action, reward, next_obs):
        self.q_table[obs, action] = (1 - self.alpha) * (self.q_table[obs, action]) + self.alpha * (reward + self.gamma * np.max(self.q_table[next_obs]))

num_episodes = 1

grid_size = 8
num_agents = 2

env = gym.make(f'Foraging-{grid_size}x{grid_size}-{num_agents}p-2f-coop-v3', max_episode_steps=100)

agents = []

for i in range(num_agents):

    num_observations = env.observation_space[i].shape[0]
    num_actions = env.action_space[i].n

    agent = QLearner(num_observations, num_actions)

    agents.append(agent)


for episode in range(1, num_episodes + 1):
    obs, infos = env.reset()
    done = False
    while not done:
        actions = []

        for idx, ob in enumerate(obs):
            actions.append(agents[idx].act(ob))
        
        next_obs, rewards, terminated, truncated, infos = env.step(actions)

        for idx, (ob, next_ob, action, reward) in enumerate(zip(obs, next_obs, actions, rewards)):
            actions.append(agents[i].update(ob, action, reward, next_ob))

        obs = next_obs
        done = terminated or truncated