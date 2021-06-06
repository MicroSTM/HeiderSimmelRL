from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import random
import torch
from torch.autograd import Variable


def _normalized_p(p, blocked_actions=None):
    """normalize a distribution"""
    if blocked_actions:
        p[0, blocked_actions] = 0.0
    return p / sum(p[0])


class EGreedy:
    """epsilon greedy"""
    def __init__(self, action_size, init_epsilon, final_epsilon, max_exp_steps):
        self.action_size = action_size
        self.init_epsilon = self.epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.max_exp_steps = max_exp_steps
        self.steps = 0


    def reset(self):
        self.epsilon = self.init_epsilon
        self.steps = 0


    def set_zero(self):
        self.epsilon = 0.0


    def update(self):
        self.steps += 1
        self.epsilon = (self.init_epsilon - self.final_epsilon) \
                        * (1.0 - float(self.steps) / float(self.max_exp_steps)) \
                        + self.final_epsilon

        
    def sample(self, policy, cuda, blocked_actions=None):
        """sample action w.r.t. given policy"""
        u = random.uniform(0, 1)
        if u < self.epsilon:
            p = _normalized_p(np.ones((1, self.action_size)), blocked_actions)
        else:
            p = _normalized_p(policy, blocked_actions)
        if cuda:
            new_policy = Variable(torch.from_numpy(p).float().cuda())
        else:
            new_policy = Variable(torch.from_numpy(p).float())
        action = new_policy.multinomial()
        return action
