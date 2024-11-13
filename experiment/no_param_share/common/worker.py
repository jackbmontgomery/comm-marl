import numpy as np
import torch


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

        print("RolloutWorker initialized")

    def generate_episode(
        self, episode_num=None, evaluate=False, epoch_num=None, eval_epoch=None
    ):
        # lists to store whole episode info
        (
            obs_ep,
            actions_ep,
            reward_ep,
            state_ep,
            avail_actions_ep,
            actions_onehot_ep,
            terminate,
            padded,
        ) = ([], [], [], [], [], [], [], [])
        self.env.reset()
        terminated = [False] * self.n_agents
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))

        # because no parameter sharing
        for i in range(self.n_agents):
            self.agents[i].policy.init_hidden(1)

        epsilon = 0 if evaluate else self.epsilon

        while not all(terminated):
            obs = self.env.get_agent_obs()
            state = np.array(obs).flatten()
            actions, avail_actions, actions_onehot = [], [], []

            all_msgs = []
            obs_np = np.array(obs)

            if self.args.with_comm:
                for agent_id in range(self.n_agents):
                    msg_agent_i = self.agents[agent_id].get_all_messages(
                        obs_np[agent_id], last_action
                    )
                    all_msgs.append(msg_agent_i)

            for agent_id in range(self.n_agents):
                avail_action = [1] * self.n_actions  # avail actions for agent_i

                all_msgs_not_agent_i = []
                msgs_agent_i = []
                if self.args.with_comm:
                    all_msgs_not_agent_i = [
                        all_msgs[i] for i in range(self.n_agents) if i != agent_id
                    ]
                    all_msgs_not_agent_i = torch.cat(
                        [x for x in all_msgs_not_agent_i], dim=0
                    )
                    msgs_agent_i = torch.tensor(all_msgs[agent_id])

                action = self.agents[agent_id].choose_action(
                    obs_np[agent_id],
                    last_action[agent_id],
                    agent_id,
                    avail_action,
                    epsilon,
                    evaluate,
                    msg_all=all_msgs_not_agent_i,
                    msg_i=msgs_agent_i,
                )

                # generate a vector of 0s and 1s of the corresponding action; actions chosen gets 1 and rest is 0
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1

                # adds action info to corresponding lists
                actions.append(action.item())
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            _, reward, terminated, _ = self.env.step(actions)

            obs_ep.append(obs)
            state_ep.append(state)

            actions = torch.Tensor(actions)

            # need to reshape the list of actions into a vector with shape (n_agents, 1) to store in the buffer
            actions_ep.append(torch.reshape(actions, [self.n_agents, 1]))
            actions_onehot_ep.append(actions_onehot)
            avail_actions_ep.append(avail_actions)
            reward_ep.append(
                [sum(reward)]
            )  # reward returned for this env is a list with a reward for each agent, so sum
            terminate.append(
                [all(terminated)]
            )  # terminated for this env is a bool list which says if each agent reached the goal or not
            padded.append([0.0])
            episode_reward += sum(reward)
            step += 1
            if self.args.epsilon_anneal_scale == "step":
                epsilon = (
                    epsilon - self.anneal_epsilon
                    if epsilon > self.min_epsilon
                    else epsilon
                )

        # handle last obs
        obs = self.env.get_agent_obs()
        state = np.array(obs).flatten()
        obs_ep.append(obs)
        state_ep.append(state)
        o_next = obs_ep[1:]
        s_next = state_ep[1:]
        obs_ep = obs_ep[:-1]
        state_ep = state_ep[:-1]

        # last obs needs to calculate the avail_actions sep, then calculate target_q
        avail_actions = [
            [1] * self.n_actions for _ in range(self.n_agents)
        ]  # same as above, as in this env everything is avail

        avail_actions_ep.append(avail_actions)
        avail_actions_next = avail_actions_ep[1:]
        avail_actions_ep = avail_actions_ep[:-1]

        # the generated episode must be self.episode_limit long, so if it terminated before this size it has to be filled, everything is filled with 1's
        for i in range(step, self.episode_limit):
            obs_ep.append(np.zeros((self.n_agents, self.obs_shape)))
            actions_ep.append(np.zeros([self.n_agents, 1]))
            state_ep.append(np.zeros(self.state_shape))
            reward_ep.append([0.0])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            actions_onehot_ep.append(np.zeros((self.n_agents, self.n_actions)))
            avail_actions_ep.append(np.zeros((self.n_agents, self.n_actions)))
            avail_actions_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.0])
            terminate.append([1.0])

        # create episode batch to add to buffer
        episode = dict(
            obs=obs_ep.copy(),
            state=state_ep.copy(),
            actions=actions_ep.copy(),
            reward=reward_ep.copy(),
            avail_actions=avail_actions_ep.copy(),
            obs_next=o_next.copy(),
            state_next=s_next.copy(),
            avail_actions_next=avail_actions_next.copy(),
            actions_onehot=actions_onehot_ep.copy(),
            padded=padded.copy(),
            terminated=terminate.copy(),
        )

        # add extra episode dimension to the dict values
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon

        return episode, episode_reward, {"steps_taken": step}
