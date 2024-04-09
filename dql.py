import math
import random
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim

from models import DQN, SimpleDQN, ReplayMemory
from envs import BaseEnv
from utils import Transition, hard_update, soft_update
from device_toolkit import set_logging


class DQL(object):

    def __init__(self, env: BaseEnv, args, writer) -> None:

        self.set_seed(args.seed)
        self.logger = set_logging(name=os.path.join(args.log_dir, 'log'))
        
        num_episodes = args.num_episodes
        self.slimmable = args.slimmable
        self.batch_size = args.batch_size
        self.validate = args.validate
        self.gamma = args.gamma
        self.eps_end = args.eps_end
        self.eps_start = args.eps_start
        self.eps_decay = args.eps_decay
        self.steps_done = 0
        self.cd_step = 0
        self.tau = args.tau
        self.env = env
        self.writer = writer
        self.optimized = False

        learning_rate = 0.01

        if args.slimmable:
            self.policy_net = DQN(num_state=[5, 6], num_actions=env.action_space.n, dropout=0.1).cuda()
            self.target_net = DQN(num_state=[5, 6], num_actions=env.action_space.n, dropout=0.1).cuda()
            self.memory = [ReplayMemory(args.buffer_size), ReplayMemory(args.buffer_size)]
        else:
            self.policy_net = SimpleDQN(num_state=6, num_actions=env.action_space.n, dropout=0.1).cuda()
            self.target_net = SimpleDQN(num_state=6, num_actions=env.action_space.n, dropout=0.1).cuda()
            self.memory = ReplayMemory(args.buffer_size)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=num_episodes, eta_min=1e-4)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.logger.info(self.policy_net)

        if args.pretrained:
            self.logger.info(f"Load model from {args.pretrained}/weight.pt")
            assert os.path.exists(os.path.join(args.pretrained, 'weight.pt')), f"{os.path.join(args.pretrained, 'weight.pt')} not found"
            self.policy_net.load_state_dict(torch.load(os.path.join(args.pretrained, 'weight.pt'), map_location=self.device))
        hard_update(self.target_net, self.policy_net)
        self.target_net.eval()


    def optimize_model(self, stage, validate=False):
        if self.slimmable:
            memory = self.memory[stage]
        else:
            memory = self.memory
        if len(memory) < self.batch_size or validate:
            return None, None
        self.optimized = True

        self.policy_net.train()

        transitions = memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
    
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None], dim=0)
        state_batch = torch.cat(batch.state, dim=0)
        action_batch = torch.cat(batch.action, dim=0)
        reward_batch = torch.cat(batch.reward, dim=0)

        if self.slimmable:
            state_action_values = self.policy_net(state_batch, stage).gather(1, action_batch)
        else:
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
    
        with torch.no_grad():
            if self.slimmable:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states, int(1 - stage)).max(1)[0]
            else:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        # criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        soft_update(self.target_net, self.policy_net, self.tau)

        self.writer.add_scalar("Loss/loss", loss, self.steps_done)
        self.writer.add_scalar("Loss/Q-value", next_state_values.mean(), self.steps_done)

    @torch.no_grad()
    def select_action(self, state: torch.Tensor, stage: int):
        '''
        Return action index
        '''
        # select a random action wih probability eps
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        # cd_threshold = 0.5 + 0.5 * math.exp(-1. * self.cd_step / self.eps_decay)
        lb = 0.5
        ub = 1
        cd_threshold = lb + (ub - lb) * (1 + math.cos((self.cd_step) * math.pi / self.eps_decay)) / 2

        if not self.validate:
            self.writer.add_scalar('Thres/eps', eps_threshold, self.steps_done)
            self.writer.add_scalar('Thres/cd', cd_threshold, self.steps_done)

        self.steps_done += 1
        # stage [cpu_freq, cpu_temp, gpu_freq, gpu_temp, time, proposal] 1x6
        cpu_freq, cpu_temp, gpu_freq, gpu_temp = state[:, :4].squeeze()
        overheat = cpu_temp < 0 or gpu_temp < 0
        num_cpu_freq = len(self.env.metadata["freq"]["cpu_freq"])
        num_gpu_freq = len(self.env.metadata["freq"]["gpu_freq"])
        if random.random() > eps_threshold or self.validate:
            self.policy_net.eval()
            if self.slimmable:
                logits = self.policy_net(state, stage)
            else:
                logits = self.policy_net(state)
            if not self.validate:
                if overheat and random.random() < cd_threshold:
                    self.cd_step += 1 if self.cd_step < self.eps_decay else 0
                    target = [tar_gpu * num_cpu_freq + tar_cpu for tar_gpu in range(int(gpu_freq) + 1) for tar_cpu in range(int(cpu_freq) + 1)]
                    mask = torch.tensor([True] * num_gpu_freq * num_cpu_freq)
                    mask[target] = False
                    logits.masked_fill_(mask.cuda(), -torch.inf)
            action = logits.max(1)[1].view(1, 1)
        else:
            if overheat:
                self.cd_step += 1 if self.cd_step < self.eps_decay else 0
                target = [tar_gpu * num_cpu_freq + tar_cpu for tar_gpu in range(int(gpu_freq) + 1) for tar_cpu in range(int(cpu_freq) + 1)]
                action = torch.tensor([random.sample(target, 1)], dtype=torch.long).cuda()
            else:
                action = torch.tensor([[self.env.random_action()]], dtype=torch.long).cuda()
    
        return action
    

    def save_checkpoint(self, path):
        torch.save(self.policy_net.state_dict(), os.path.join(path, 'weight.pt'))


    def scheduler_step(self):
        if not self.optimized:
            return
        if self.scheduler is not None:
            self.scheduler.step()

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)


    