from collections import namedtuple, deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from slimmable_ops import SlimmableLinear, SlimmableBlock, SlimmableBatchNorm1d
from utils import Transition


input_feat_norm = SlimmableBatchNorm1d

class DQN(nn.Module):

    def __init__(self, num_state, num_actions=100, dropout=0.1):
        super(DQN, self).__init__()
        
        assert isinstance(num_state, list), f"Got {num_state}"
        num_feat = 64
        scale = min(num_state) / max(num_state)

        self.embedding = nn.Embedding(num_embeddings=2, embedding_dim=max(num_state))
        
        self.input_norm = input_feat_norm(num_features_list=num_state)

        self.layer1 = SlimmableBlock(in_features_list=num_state,
                                     out_features_list=[num_feat, num_feat],
                                     act=nn.ReLU(True),
                                     dropout=dropout)
        
        self.layer2 = LinearBlock(in_features=num_feat,
                                  out_features=2 * num_feat,
                                  act=nn.ReLU(True),
                                  dropout=dropout)

        self.layer3 = LinearBlock(in_features=2 * num_feat,
                                  out_features=4 * num_feat,
                                  act=nn.ReLU(True),
                                  dropout=dropout)

        self.last_layer = nn.Linear(in_features=4 * num_feat,
                                    out_features=num_actions)

        # self.layer1 = SlimmableBlock(in_features_list=num_state,
        #                              out_features_list=[num_feat, int(num_feat * scale)],
        #                              act=nn.ReLU(True),
        #                              dropout=dropout)
        
        # self.layer2 = SlimmableBlock(in_features_list=[num_feat, int(num_feat * scale)],
        #                              out_features_list=[2 * num_feat, int(2 * num_feat)],
        #                              act=nn.ReLU(True),
        #                              dropout=dropout)
        
        # self.layer3 = SlimmableBlock(in_features_list=[2 * num_feat, int(2 * num_feat * scale)],
        #                              out_features_list=[4 * num_feat, int(4 * num_feat * scale)],
        #                              act=nn.ReLU(True),
        #                              dropout=dropout)
        
        
        # self.last_layer = SlimmableLinear(in_features_list=[4 * num_feat, int(4 * num_feat * scale)],
        #                                   out_features_list=[num_actions, num_actions])
        
        self._weight_init()

    def forward(self, obs: torch.Tensor, stage: int):
        
        b, n_state = obs.size()
        # _stage = torch.LongTensor([stage] * b).to(obs.device)
        _stage = torch.LongTensor([stage] * b).to(obs.device)
        emb_stage = self.embedding(_stage)[:, :n_state]

        emb_obs = self.input_norm(obs, stage) + emb_stage

        feat = self.layer1(emb_obs, stage)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        action = self.last_layer(feat)
        # feat = self.layer2(feat, stage)
        # feat = self.layer3(feat, stage)
        # action = self.last_layer(feat, stage)
        
        return action

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in')

class Actor(DQN):

    def __init__(self, num_state, num_actions=100, dropout=0.1):
        super(Actor, self).__init__(num_state, num_actions, dropout)
        
    def forward(self, obs: torch.Tensor, stage: int):
        
        return super().forward(obs, stage).softmax(dim=-1)


class Critic(DQN):

    def __init__(self, num_state, num_actions=100, dropout=0.1):
        super(Critic, self).__init__(num_state, num_actions, dropout)
        num_feat = 64
        scale = min(num_state) / max(num_state)

        self.action_embed = nn.Embedding(num_embeddings=num_actions, embedding_dim=num_feat)

        self.layer2 = LinearBlock(in_features=num_feat,
                                  out_features=2 * num_feat,
                                  act=nn.ReLU(True),
                                  dropout=dropout)

        self.last_layer = nn.Linear(in_features=4 * num_feat, out_features=1)

        # self.layer2 = SlimmableBlock(in_features_list=[int(num_feat) + num_actions, int(num_feat * scale) + num_actions],
        #                              out_features_list=[2 * num_feat, int(2 * num_feat * scale)],
        #                              act=nn.ReLU(True),
        #                              dropout=dropout)

        # self.last_layer = SlimmableLinear(in_features_list=[4 * num_feat, int(4 * num_feat * scale)], out_features_list=[1, 1])

        self._weight_init()


    def forward(self, obs: torch.Tensor, action: torch.Tensor, stage: int):
        b, n_state = obs.size()
        # _stage = torch.LongTensor([stage] * b).to(obs.device)
        _stage = torch.LongTensor([stage] * b).to(obs.device)
        emb_stage = self.embedding(_stage)[:, :n_state]

        emb_obs = self.input_norm(obs, stage) + emb_stage

        feat = self.layer1(emb_obs, stage)
        emb_action = self.action_embed(action.long()).squeeze()
        # concat_feat = torch.cat([feat, action], dim=-1)

        # feat = self.layer2(concat_feat, stage)
        # feat = self.layer3(feat, stage)
        # score = self.last_layer(feat, stage)
        feat = self.layer2(feat + emb_action)
        feat = self.layer3(feat)
        score = self.last_layer(feat)
        return score


class SimpleDQN(nn.Module):

    def __init__(self, num_state, num_actions=156, dropout=0.1):
        super(SimpleDQN, self).__init__()
        
        num_feat = 64
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=num_state)
        self.layer1 = LinearBlock(in_features=num_state,
                                  out_features=int(num_feat),
                                  act=nn.ReLU(True),
                                  dropout=dropout)
        
        self.layer2 = LinearBlock(in_features=int(num_feat),
                                  out_features=int(2 * num_feat),
                                  act=nn.ReLU(True),
                                  dropout=dropout)
        
        self.layer3 = LinearBlock(in_features=int(2 * num_feat),
                                  out_features=int(4 * num_feat),
                                  act=nn.ReLU(True),
                                  dropout=dropout)
        
        
        self.last_layer = nn.Linear(in_features=int(4 * num_feat), out_features=num_actions)
        
        self._weight_init()

    def forward(self, obs: torch.Tensor):

        # n_state = obs.size(-1)
        stage, state = obs[:, 0], obs[:, 1:]
        emb_stage = self.embedding(stage.long().squeeze())
        emb_obs = state + emb_stage
        feat = self.layer1(emb_obs)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        action = self.last_layer(feat)
        
        return action

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m, 'weight_orig'):
                    nn.init.kaiming_normal_(m.weight_orig)
                    m.weight_orig.data *= 0.1
                else:
                    nn.init.kaiming_normal_(m.weight)

class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, act=nn.ReLU(True), dropout=0.1) -> None:
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.norm = nn.BatchNorm1d(num_features=out_features)
        self.activation = act
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    
    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))




class ReplayMemory(object):

    def __init__(self, capacity, stage=0, ratio=0.5):
        self.capacity = capacity
        self.positive_memory = deque([],maxlen=int(capacity * ratio))
        self.pos_len = int(capacity * ratio)
        self.negtive_memory = deque([],maxlen=capacity - int(capacity * ratio))
        self.neg_len = capacity - self.pos_len
        self.stage = stage

    def push(self, **kwargs):
        """Save a transition"""
        if kwargs.get('reward') < 0:                
            self.negtive_memory.append(Transition(**kwargs))
        else:
            self.positive_memory.append(Transition(**kwargs))

    def sample(self, batch_size):
        transitions = random.sample(list(self.positive_memory) + list(self.negtive_memory), batch_size)
        return transitions

    def __len__(self):
        return len(self.positive_memory) + len(self.negtive_memory)