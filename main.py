from pettingzoo.mpe import simple_spread_v3
import torch
import numpy as np

# Initialize the PettingZoo simple_spread environment
env = simple_spread_v3.env(
    N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False
)
env.reset()

# Fetch environment properties
n_agents = len(env.possible_agents)

# observation_space = env.observation_space(env.agents[0])
# action_space = env.action_spaces[z]

# Define the agent's configuration based on observations and actions from the environment
n_s_ls = [18] * n_agents  # Observation dimensions per agent
n_a_ls = [5] * n_agents  # Action dimensions per agent
neighbor_mask = np.eye(n_agents)  # Simple identity for neighbors, can be adapted

# Define other necessary parameters
distance_mask = np.zeros(
    (n_agents, n_agents)
)  # No distance metric in this simple setup
coop_gamma = 0.99
total_step = 1000  # Arbitrary total steps, adjust as necessary
seed = 42

# Instantiate the MA2C_DIAL agent with hardcoded config
model_config = {
    "rmsp_alpha": 0.99,
    "rmsp_epsilon": 1e-5,
    "max_grad_norm": 40,
    "gamma": 0.99,
    "lr_init": 5e-4,
    "lr_decay": "constant",
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "num_lstm": 64,
    "num_fc": 64,
    "batch_size": 120,
    "reward_norm": 2000.0,
    "reward_clip": -1,
}

agent = MA2C_DIAL(
    n_s_ls=n_s_ls,
    n_a_ls=n_a_ls,
    neighbor_mask=neighbor_mask,
    distance_mask=distance_mask,
    coop_gamma=coop_gamma,
    total_step=total_step,
    model_config=model_config,
    seed=seed,
    use_gpu=False,  # Set to True if using GPU
)

# Set up for training
num_episodes = 10  # Number of episodes for testing
for episode in range(num_episodes):
    env.reset()
    done = {agent: False for agent in env.agents}
    observations = {agent: env.observe(agent) for agent in env.agents}

    while not all(done.values()):
        # Convert observations to torch tensors and select actions
        actions = {}
        for i, agent_name in enumerate(env.agents):
            if not done[agent_name]:
                obs_tensor = torch.tensor(
                    observations[agent_name], dtype=torch.float32
                ).unsqueeze(0)
                action_probs = agent._init_policy().act(obs_tensor)
                action = action_probs.argmax().item()
                actions[agent_name] = action

        # Step in the environment
        next_observations, rewards, dones, infos = {}, {}, {}, {}
        for agent_name in env.agents:
            if not done[agent_name]:
                obs, reward, done, info = env.step(actions[agent_name])
                next_observations[agent_name] = obs
                rewards[agent_name] = reward
                dones[agent_name] = done
                infos[agent_name] = info

        # Set the next observations and done flags
        observations = next_observations
        done.update(dones)

env.close()
