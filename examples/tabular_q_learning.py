import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
import seaborn as sns

def plot_q_table(q_table):
    """
    Visualize the Q-table by color coding the states based on their maximum Q-values.

    Parameters:
    q_table (np.ndarray): Q-table with shape (num_states, num_actions)
    """
    # Compute the value of each state as the maximum Q-value across actions
    state_values = np.max(q_table, axis=1)
    
    # Reshape state values into a 4x4 grid for Frozen Lake (if applicable)
    grid_size = int(np.sqrt(state_values.size))
    state_values_grid = state_values.reshape((grid_size, grid_size))
    
    # Plot the heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(state_values_grid, annot=True, cmap='coolwarm', cbar=True, square=True, 
                xticklabels=False, yticklabels=False, cbar_kws={'label': 'State Value'})
    
    plt.title("State-Value Heatmap based on Maximum Q-values")
    plt.show()

env = gym.make("FrozenLake-v1", is_slippery=False, max_episode_steps=100)
num_episodes = 500
gamma = 0.99
eps = 1
eps_decay = 0.9999
eps_min = 0.1
alpha = 1

num_actions = env.action_space.n
observation_space_size = env.observation_space.n

q_table = np.zeros(shape=(observation_space_size, num_actions))

for episode in range(num_episodes):
    done = False
    total_rewards = 0
    obs, info = env.reset(seed=2024)

    while not done:

        if random.random() < eps:
            action = env.action_space.sample()

        else:
            actions_values = q_table[obs, :]
            action = np.argmax(actions_values)

        new_obs, reward, terminated, truncated, info = env.step(action)

        # Bell-man Equation

        q_table[obs, action] = (1 - alpha) * (q_table[obs, action]) + alpha * (reward + gamma * np.max(q_table[new_obs]))

        total_rewards += reward
        eps = max(eps * eps_decay, eps_min)
        done = terminated or truncated

        obs = new_obs

done = False
total_rewards = 0
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode='rgb_array')
obs, info = env.reset(seed=2024)

while not done:

    actions_values = q_table[obs]

    action = np.argmax(actions_values)

    new_obs, reward, done, truncated, info = env.step(action)

    # Bell-man Equation

    total_rewards += reward
    eps = max(eps * eps_decay, eps_min)

    obs = new_obs