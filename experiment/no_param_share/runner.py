from agent.agent import Agents
from common.worker import RolloutWorker
from common.buffer import ReplayBuffer
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import wandb


class Runner:
    def __init__(self, env, args):
        self.env = env

        # no parameter sharing: each agent is independent
        self.list_agents = [Agents(args) for i in range(args.n_agents)]

        self.rolloutWorker = RolloutWorker(env, self.list_agents, args)

        self.buffer = ReplayBuffer(args)

        self.args = args

        self.save_path = self.args.result_dir + "/" + args.alg
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):

        wandb.init(project="comm-marl", config=self.args)

        plt.figure()
        plt.axis([0, self.args.n_steps, 0, 100])
        win_rates = []
        episode_rewards = []
        train_steps = 0
        time_steps = 0
        evaluate_steps = -1

        start_time = time.time()

        while time_steps < self.args.n_steps:
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                elapsed_time = time.time() - start_time
                print(
                    "Run {}, train step {}/{}, elapsed time: {:.2f} seconds".format(
                        num, time_steps, self.args.n_steps, elapsed_time
                    )
                )

                episode_reward = self.evaluate(epoch_num=time_steps)

                wandb.log({"reward": episode_reward, "time_steps": time_steps})

                episode_rewards.append(episode_reward)

                plt.plot(range(len(episode_rewards)), episode_rewards)
                plt.xlabel("step*{}".format(self.args.evaluate_cycle))
                plt.ylabel("episode_rewards")

                plt.savefig(
                    self.save_path
                    + "/plt_{}_{}_{}ts.png".format(
                        num, self.args.env, self.args.n_steps
                    ),
                    format="png",
                )
                np.save(
                    self.save_path
                    + "/episode_rewards_{}_{}_{}ts".format(
                        num, self.args.env, self.args.n_steps
                    ),
                    episode_rewards,
                )
                evaluate_steps += 1

            episodes = []

            for episode_idx in range(self.args.n_episodes):
                episode, _, info = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += info["steps_taken"]

            episode_batch = episodes[0]
            episodes.pop(0)

            # put observations of all the generated epsiodes together
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate(
                        (episode_batch[key], episode[key]), axis=0
                    )

            # again, coma doesnt need buffer, so wont store the episodes sampled
            if self.args.alg.find("coma") > -1:
                episode_batch["terminated"] = episode_batch["terminated"].astype(float)

                self.agents.train(
                    episode_batch, train_steps, self.rolloutWorker.epsilon
                )
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(
                        min(self.buffer.current_size, self.args.batch_size)
                    )
                    # first gather the messages from all the agents
                    all_msgs, all_msgs_next = [], []
                    if self.args.with_comm:
                        for agent_id in range(self.args.n_agents):
                            msgs_i, msgs_i_next = self.list_agents[
                                agent_id
                            ].get_msgs_for_train(mini_batch, train_steps, agent_id)
                            all_msgs.append(msgs_i)
                            all_msgs_next.append(msgs_i_next)

                    for agent_id in range(self.args.n_agents):
                        self.list_agents[agent_id].train(
                            mini_batch,
                            train_steps,
                            agent_id,
                            all_msgs=all_msgs,
                            all_msgs_next=all_msgs_next,
                        )

                    train_steps += 1

        wandb.finish()
        # saving final trained networks
        for agent_id in range(self.args.n_agents):
            self.list_agents[agent_id].policy.save_model(
                train_steps, agent_id, end_training=True
            )

        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(win_rates)), win_rates)
        plt.xlabel("steps*{}".format(self.args.evaluate_cycle))
        plt.ylabel("win_rate")

        plt.subplot(2, 1, 2)
        plt.plot(range(len(episode_rewards)), episode_rewards)
        plt.xlabel("steps*{}".format(self.args.evaluate_cycle))
        plt.ylabel("episode_rewards")

        plt.savefig(
            self.save_path
            + "/plt_{}_{}_{}ts.png".format(num, self.args.env, self.args.n_steps),
            format="png",
        )
        np.save(
            self.save_path
            + "/episode_rewards_{}_{}_{}ts".format(
                num, self.args.env, self.args.n_steps
            ),
            episode_rewards,
        )
        np.save(
            self.save_path
            + "/win_rates_{}_{}_{}ts".format(num, self.args.env, self.args.n_steps),
            win_rates,
        )

    def evaluate(self, epoch_num=None):
        episode_rewards = 0
        steps_avrg = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, info = self.rolloutWorker.generate_episode(
                epoch, evaluate=True, epoch_num=epoch_num
            )  # , epoch_num=epoch_num, eval_epoch=epoch) # changed added this
            episode_rewards += episode_reward
            steps_avrg += info["steps_taken"]

        return episode_rewards / self.args.evaluate_epoch
