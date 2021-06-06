from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
from pathlib import Path
import sys
import random
import time
import math
import pickle
import visdom
import importlib

import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim

from models import ActorCriticCNN
from sampler import EGreedy
from envs.box2d import *
from utils import *


def _str2class(str):
    """convert string to class name"""
    return getattr(sys.modules[__name__], str)


class A2C_2agents:
    """
    A2C Agents
    """
    def __init__(self, args):
        self.args = args

        # random seed
        random.seed(args.seed)

        # specify environment
        self.env = _str2class(args.exp_name)(random_pos=args.random_pos, restitution=args.restitution)

        self.action_size = self.env.action_size
        self.num_agents = 2

        # build model
        self.models = [ActorCriticCNN(self.env.obs_dim, self.env.action_size)
                        for _ in range(self.num_agents)]

        if self.args.cuda:
            [model.cuda() for model in self.models]

        self.checkpoint_dir_all = ["".join([args.checkpoint_dir, '/', args.exp_name, "/agent", str(agent_id)]) 
                                for agent_id in range(self.num_agents)]
        for checkpoint_dir in self.checkpoint_dir_all:
            p = Path(checkpoint_dir)
            if not p.is_dir():
                p.mkdir(parents = True)
        self.record_dir = args.record_dir + '/' + args.exp_name 
        p = Path(self.record_dir)
        if not p.is_dir():
            p.mkdir(parents = True)

        self.sampler = EGreedy(self.action_size, 
                               args.init_epsilon, 
                               args.final_epsilon, 
                               args.max_exp_steps)


    def select_action(self, model, state, hidden_state):
        """select action"""
        if self.args.cuda:
            state = Variable(torch.from_numpy(state).float().unsqueeze(0).cuda())
        else:
            state = Variable(torch.from_numpy(state).float().unsqueeze(0))

        policy, value, hidden_state \
            = model(state, hidden_state)
        action = self.sampler.sample(policy.cpu().data.numpy(), self.args.cuda)
        if self.args.verbose > 1:
            print('policy:', policy.cpu().data.numpy()[0])
            print('action:', self.env.action_space[int(action.data.cpu().numpy()[0][0])])

        return state.cpu().data, \
               int(action.data.cpu().numpy()[0][0]), \
               policy.cpu().data, value, \
               hidden_state


    def act(self, state_all, hx_all, cx_all):
        """act one step w.r.t. policy for all agents"""
        if self.args.verbose > 1:
            print('step ', self.env.steps)
        action_all, policy_all, value_all = [], [], []
        for agent_id in range(self.num_agents):
            if self.args.exp_name != 'Blocking_v0' or agent_id: # Blocking_v0 is for HO scenarios
                state_all[agent_id], action, policy, value, (hx_all[agent_id], cx_all[agent_id]) = \
                    self.select_action(self.models[agent_id], 
                                       state_all[agent_id], 
                                       (hx_all[agent_id], cx_all[agent_id]))
                if self.env.steps % self.args.action_freq == 0:
                    self.env.send_action(agent_id, self.env.action_space[action])
            else:
                 state_all[agent_id], action, policy, value, hx_all[agent_id], cx_all[agent_id] = \
                 None, None, None, None, None, None
            action_all.append(action)
            policy_all.append(policy) 
            value_all.append(value)
        self.env.step()
        current_r_all = self.env.get_reward()
        return state_all, action_all, policy_all, value_all, (hx_all, cx_all), current_r_all


    def rollout(self, record=False):
        """rollout for an episode"""
        if self.args.cuda:
            hx_all = [Variable(torch.zeros(1, model.latent_dim).cuda()) for model in self.models]
            cx_all = [Variable(torch.zeros(1, model.latent_dim).cuda()) for model in self.models]
        else:
            hx_all = [Variable(torch.zeros(1, model.latent_dim)) for model in self.models]
            cx_all = [Variable(torch.zeros(1, model.latent_dim)) for model in self.models]
        nb_steps = 0
        c_r_all = [0] * self.num_agents

        if record:
            self.env.setup(
                str(Path(self.video_dir, '{0}.mp4'.format(self.episode_id))))
        else:
            self.env.setup()
        self.env.start()

        while self.env.running:
            state_all = self.env.get_obs()
            state_all, action_all, policy_all, value_all, (hx_all, cx_all), current_r_all \
                    = self.act(state_all, hx_all, cx_all)
            nb_steps += 1
            c_r_all  = [c_r + current_r for c_r, current_r in zip(c_r_all, current_r_all)]
            if nb_steps >= self.args.max_episode_length:
                break

        self.env.release()

        if record:
            return c_r_all, self.env.trajectories
        else:
            return c_r_all


    def test(self, checkpoint_list=0):
        """test the model"""
        args = self.args
        nb_episodes = args.nb_episodes
        for agent_id in range(self.num_agents):
            self.models[agent_id].eval()
        for agent_id, checkpoint in enumerate(checkpoint_list):
            if checkpoint == 0: continue
            self.load_model(self.models[agent_id],
                            self.checkpoint_dir_all[agent_id] + "/checkpoint_" + \
                                        str(checkpoint))
        p = Path(self.record_dir, '{}_{}{}'.format(checkpoint_list[0], 
                                                   checkpoint_list[1],
                                                   '_' + str(args.restitution) if args.restitution > 0 else '') , str(int(100 / self.args.action_freq + 0.5)))
        if not p.is_dir():
            p.mkdir(parents = True)
        self.video_dir = str(p)
        self.sampler.reset() 
        self.sampler.set_zero()           
        cumulative_rewards = [None] * nb_episodes
        all_vels = []
        steps = [None] * nb_episodes

        for episode_id in range(nb_episodes):
            self.episode_id = episode_id
            c_r_all, trajectories = self.rollout(record=True)
            print("episode: #{} steps: {} reward: {}".format(episode_id, self.env.steps, c_r_all[0]))
            cumulative_rewards[episode_id] = c_r_all[0]
            steps[episode_id] = self.env.steps
            # pickle.dump({'trajectories': trajectories, 'reward': rewards, 'Vret': Vret, 'Vs': Vs}, 
            #             open(self.video_dir + "/{0}.pik".format(episode_id), "wb"))
            cur_ave_vel = get_ave_vel(trajectories)
            print("average velocities:", cur_ave_vel)
            all_vels += cur_ave_vel
            file = open(str(p / '{0}.txt'.format(episode_id)), 'w')
            for t in range(len(trajectories[0])):
                file.write('%.3f %.3f %.3f %.3f\n' \
                    % (trajectories[0][t][0], trajectories[0][t][1], 
                       trajectories[1][t][0], trajectories[1][t][1]))

        print("ave reward:", sum(cumulative_rewards) / len(cumulative_rewards))
        print("ave velocities:", sum(all_vels) / len(all_vels))
        print("ave steps:", sum(steps) / len(steps))


    def load_model(self, model, path):
        """load trained model parameters"""
        model.load_state_dict(dict(torch.load(path)))


