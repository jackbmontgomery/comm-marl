from argparse import ArgumentParser
import torch


def common_args():
    parser = ArgumentParser()

    # general args
    parser.add_argument("--env", "-e", default="TrafficJunction", help="set env name")
    parser.add_argument(
        "--n_steps",
        "-ns",
        type=int,
        default=250000,
        help="set total time steps to run",
    )
    parser.add_argument(
        "--n_episodes", "-nep", type=int, default=1, help="set n_episodes"
    )
    parser.add_argument("--epsilon", "-eps", default=0.5, help="set epsilon value")
    parser.add_argument(
        "--last_action",
        type=bool,
        default=True,
        help="whether to use the last action to choose action",
    )
    parser.add_argument(
        "--reuse_network",
        type=bool,
        default=True,
        help="whether to use one network for all agents",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor")
    parser.add_argument(
        "--evaluate_epoch",
        type=int,
        default=20,
        help="the number of the epoch to evaluate the agent",
    )
    parser.add_argument(
        "--alg", type=str, default="idql", help="the algorithm to train the agent"
    )
    parser.add_argument("--optimizer", type=str, default="RMS", help="the optimizer")
    parser.add_argument(
        "--model_dir", type=str, default="./model", help="model directory of the policy"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./result",
        help="result directory of the policy",
    )
    parser.add_argument(
        "--load_model",
        type=bool,
        default=False,
        help="whether to load the pretrained model",
    )
    parser.add_argument(
        "--learn", type=bool, default=True, help="whether to train the model"
    )
    parser.add_argument(
        "--evaluate_cycle", type=int, default=10000, help="how often to eval the model"
    )
    parser.add_argument(
        "--target_update_cycle",
        type=int,
        default=200,
        help="how often to update the target network",
    )
    parser.add_argument(
        "--save_cycle", type=int, default=6650, help="how often to save the model"
    )

    # if doing communication
    parser.add_argument(
        "--with_comm", type=int, default=1, help="whether to use communication"
    )
    parser.add_argument(
        "--msg_cut", type=bool, default=False, help="whether to cut msg"
    )
    parser.add_argument("--cuda_device", type=int, default=0, help="which gpu number")
    parser.add_argument(
        "--rnn_hidden_dim", type=int, default=64, help="which rnn hidden dim"
    )

    parser.add_argument(
        "--parameter_sharing", type=bool, default=False, help="Fixed for wandb logging"
    )

    args = parser.parse_args()

    args.with_comm = args.with_comm == 1

    return args


def config_args(args):
    # buffer/batch sizes
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # epsilon args
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = "step"

    # network
    args.lr = 5e-4

    args.comm_net_dim = 64

    # train steps
    args.train_steps = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # msg dim after net
    args.final_msg_dim = 10

    return args
