import gym
import ma_gym
from common.arguments import common_args, config_args
from runner import Runner

import numpy as np
import torch
import random

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

N_EXPERIMENTS = 1

if __name__ == "__main__":
    args = common_args()

    seed = random.randrange(0, 2**32 - 1)
    print("Using seed: ", seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    if args.env in ["TrafficJunction"]:
        env = gym.make("TrafficJunction4-v0")
        args.n_actions = env.action_space[0].n
        args.n_agents = env.n_agents
        args.state_shape = 81 * args.n_agents
        args.obs_shape = 81
        args.episode_limit = env._max_steps

    elif args.env in ["PredatorPrey"]:
        # to avoid registering a whole new environment just do it here for now
        env = gym.make(
            "PredatorPrey7x7-v0",
            grid_shape=(7, 7),
            n_agents=4,
            n_preys=2,
            penalty=-0.75,
        )
        args.n_actions = env.action_space[0].n
        args.n_agents = env.n_agents
        args.state_shape = 28 * args.n_agents
        args.obs_shape = 28
        args.episode_limit = env._max_steps

    else:
        raise Exception("Invalid environment: environment not supported!")

    print(
        "Environment {} initialized, for {} time steps and evaluating every {} time steps".format(
            args.env, args.n_steps, args.evaluate_cycle
        )
    )

    args = config_args(args)

    print("Communication set to", args.with_comm)
    print("With args:\n", args)

    runner = Runner(env, args)

    # parameterize run according to the number of independent experiments to run, i.e., independent sets of n_epochs over the model; default is 1
    if args.learn:
        runner.run(N_EXPERIMENTS)
