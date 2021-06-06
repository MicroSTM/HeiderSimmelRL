from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import torch

from agents import A2C_2agents

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--t-max', type=int, default=50, help='Max number of forward steps for A2C before update')
parser.add_argument('--max-episode-length', type=int, default=30, help='Maximum episode length')
parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size of LSTM cell')
parser.add_argument('--memory-capacity-episodes', type=int, default=10000, help='Experience replay memory capacity')
parser.add_argument('--replay-ratio', type=int, default=4, help='Ratio of off-policy to on-policy updates')
parser.add_argument('--replay-start', type=int, default=100, help='Number of transitions to save before starting off-policy training')
parser.add_argument('--discount', type=float, default=0.95, help='Discount factor')
parser.add_argument('--no-time-normalization', action='store_true', default=False, help='Do not normalize loss by number of time steps')
parser.add_argument('--max-gradient-norm', type=float, default=1, help='Max value of gradient L1 norm for gradient clipping')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch-size', type=int, default=64, help='Off-policy batch size')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='checkpoints directory')
parser.add_argument('--record-dir', type=str, default='./record_replay', help='record directory')
parser.add_argument('--video-dir', type=str, default='', help='video directory, empty if using defaut folder names')
parser.add_argument('--init-epsilon', type=float, default=0.1, help='Initial e-greedy coefficient')
parser.add_argument('--final-epsilon', type=float, default=0.0, help='Final e-greedy coefficient')
parser.add_argument('--checkpoint-episodes', type=int, default=1000, help='Frequency of saving checkpoints')
parser.add_argument('--max-exp-steps', type=int, default=100000, help='Maximum steps for exploration')
parser.add_argument('--neg_ratio', type=float, default=0.5, help='Ratio of negative experiences in a batch')
parser.add_argument('--exp-name', type=str, default='Blocking_v2', help='Experiment name')
parser.add_argument('--min-prob', type=float, default=1e-6, help='Minimum policy prob')
parser.add_argument('--visdom', action='store_true', default=False, help='Whether to use visdom for monitoring training')
parser.add_argument('--random-pos', action='store_true', default=False, help='Whether to sample random positions for the first entity')
parser.add_argument('--restitution', type=float, default=0, help='Restitution of the first entity')
parser.add_argument('--action-freq', type=int, default=1, help='Action frequency')
parser.add_argument('--checkpoints', nargs='*', type=int, default=[0, 0], help='checkpoints to be loaded')
parser.add_argument('--trainable-agents', nargs='*', type=int, default=[0], help='Trainable agents')
parser.add_argument('--nb-episodes', type=int, default=100, help='Number of episodes')
parser.add_argument('--verbose', type=int, default=1, help='How much info to display')


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print (' ' * 20 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 20 + k + ': ' + str(v))

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    agent = A2C_2agents(args)
    agent.test(args.checkpoints)
    # agent.test([100000, 16000])
    # agent.test([34400, 16000])
    # agent.test([34400, 1000])
