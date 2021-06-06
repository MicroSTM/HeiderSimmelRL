from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from utils import weights_init


class ActorCriticCNN(torch.nn.Module):
    """
    actor-critic net with CNN
    """
    def __init__(self,
                 input_dim, 
                 action_size,  
                 latent_dim = 256,
                 LSTM = True
                 ):
        super(ActorCriticCNN, self).__init__()

        self.action_size = action_size
        self.input_dim = input_dim
        self.state_size = input_dim[0] * input_dim[1] * input_dim[2]
        self.latent_dim = latent_dim
        self.LSTM = LSTM

        self.conv1 = nn.Conv2d(input_dim[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        if LSTM:
            self.lstm = nn.LSTMCell(64 * 7 * 7, self.latent_dim)
        else:
            self.fc = nn.Linear(64 * 7 * 7, self.latent_dim)
        self.critic_linear = nn.Linear(self.latent_dim, 1)
        self.actor_linear = nn.Linear(self.latent_dim, self.action_size)

        # init weights
        self.apply(weights_init)
        if LSTM:
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)

    
    def forward(self, 
                state, 
                hidden = None):
        state = state.contiguous().view(-1, self.input_dim[0], 
                                            self.input_dim[1], 
                                            self.input_dim[2])
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 7 * 7)

        if self.LSTM:
            (hx, cx) = self.lstm(x, hidden)
            x = hx
        else:
            x = F.relu(self.fc(x))
        policy = F.softmax(self.actor_linear(x), dim=-1)
        V = self.critic_linear(x)

        if self.LSTM:
            return policy, V, (hx, cx)
        else:
            return policy, V

