from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import random
import math
import sys
import pickle
from pathlib import Path

from utils import *
from envs.box2d import *


def _str2class(str):
    return getattr(sys.modules[__name__], str)


parser = argparse.ArgumentParser()
parser.add_argument('--max-episode-length', type=int, default=500, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--record-dir', type=str, default='./record_replay', metavar='RECORD', help='Mission record directory')
parser.add_argument('--exp-name', type=str, default='collision', help='Experiment name')
parser.add_argument('--nb-episodes', type=int, default=100, help='Number of episodes')


def gen(args):
    """generate physical interactions"""
    num_episodes = 100
    env = _str2class(args.exp_name)()
    all_vels = []
    for episode_id in range(num_episodes):
        f_combined = 1200.0 / 2
        angles = [random.uniform(0, 2) * math.pi for _ in range(2)]
        forces = [(f_combined * math.cos(angle), 
                   f_combined * math.sin(angle)) for angle in angles]
        
        p = Path(args.record_dir, args.exp_name)
        if not p.is_dir():
            p.mkdir(parents=True)
        env.setup([1, 1], [1, 1], str(p / '{0}.mp4'.format(episode_id)))
        env.start()
        env.apply_force(0, *forces[0])
        env.apply_force(1, *forces[1])
        env.step()
        
        nb_steps = 0
        while True:
            nb_steps += 1
            env.step()
            if nb_steps >= args.max_episode_length:
                break
        file = open(str(p / '{0}.txt'.format(episode_id)), 'w')
        for t in range(args.max_episode_length):
            file.write('%.3f %.3f %.3f %.3f\n' % (env.trajectories[0][t][0], env.trajectories[0][t][1], env.trajectories[1][t][0], env.trajectories[1][t][1]))
        cur_ave_vel = get_ave_vel(env.trajectories)
        print("average velocities:", cur_ave_vel)
        all_vels += cur_ave_vel
        env.release()
    print("ave velocities:", sum(all_vels) / len(all_vels))


if __name__ == '__main__':
    args = parser.parse_args()
    print (' ' * 20 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 20 + k + ': ' + str(v))
    gen(args)
