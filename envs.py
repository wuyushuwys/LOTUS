from collections import deque
import random
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from device_toolkit import TX2CPU, TX2GPU, OrinNanoCPU, OrinNanoGPU
from device_toolkit import Server, PORT


class BaseEnv(gym.Env):

    def __init__(self,
                 max_steps=100,
                 throttling=None,
                 seed=0,
                 deadline=400,
                 penalty=2,
                 beta=1,
                 random_throttling=False,
                 gpu_freq=None,
                 cpu_freq=None,
                 log_name='logs/base_server',
                 slimmable=False,
                 deadline_split=1,
                 temp_control=False,
                 fan=0) -> None:
        super(BaseEnv, self).__init__()

        self.metadata = {
            "vectorize": {"gpu_freq": len(gpu_freq), "cpu_freq": len(cpu_freq)},
            "freq2id": {"gpu_freq": {v:i for i, v in enumerate(gpu_freq)}, "cpu_freq": {v:i for i, v in enumerate(cpu_freq)}},
            "freq": {"gpu_freq": gpu_freq, "cpu_freq": cpu_freq},
        }
        
        self._max_steps = max_steps
        self._throttling = throttling/1000
        self._throttling_bound = self._throttling if throttling else 99.5
        assert 50 <= self._throttling_bound <= 99.5
        self.seed(seed)
        self.deadline=deadline
        self.penalty = torch.tensor(penalty, dtype=torch.float)
        self.beta = beta
        self.random_throttling = random_throttling
        self.temp_control = 1 if temp_control else 0

        self.socket_server = Server(address='', port=PORT, name=log_name)
        self._init_client(throttling=None, fan=fan)

        self.action_space = spaces.Discrete(len(gpu_freq) * len(cpu_freq))

        self._step_time = 0
        self._num_steps = 0
        self._time_list = []
        self._rewards = [[], [], [], []]
        self._accumulated_reward = 0
        self.sync_t = 1  # num of inference value as in device 
        self._action_to_frequency = {len(cpu_freq) * i_g + i_c: (f_g, f_c)
                                     for i_g, f_g in enumerate(gpu_freq) for i_c, f_c in enumerate(cpu_freq)}
        
        self.slimmable = slimmable
        self.deadline_split = deadline_split
        self.previous_cpu_thres = 0
        self.previous_gpu_thres = 0
        self.stage1_buffer = deque([],maxlen=3)
        self.stage2_buffer = deque([],maxlen=3)
        # self.reset()

    def _init_client(self, throttling=80000, fan=None):
        spec = dict()
        spec['gov'] = 'userspace'
        # if throttling:
        #     spec['throttling'] = throttling
        sending_message = dict(
            CPU=dict(**spec, freq=self.metadata['freq']['cpu_freq'][0]),
            GPU=dict(**spec, freq=self.metadata['freq']['gpu_freq'][0]),
            FAN = dict(control=self.temp_control, speed=fan if fan else 0)
        )
        self.socket_server.send(sending_message)

    def _get_obs(self):
        recv_message = self.socket_server.recv()

        if isinstance(recv_message, dict):
            stage = recv_message['stage']
            
            cpu_freq = recv_message['specs']['CPU']['freq']['0']  # scale to GHz
            cpu_freq =  self.metadata['freq2id']['cpu_freq'][cpu_freq]
            
            cpu_temp = recv_message['specs']['CPU']['temp']/1000  # scale micro-celsius to celsius
            
            gpu_freq = recv_message['specs']['GPU']['freq']  # scale to GHz
            gpu_freq = self.metadata['freq2id']['gpu_freq'][gpu_freq]
            
            gpu_temp = recv_message['specs']['GPU']['temp']/1000  # scale micro-celsius to celsius

            power = recv_message['specs']['POWER']['current']/1000 * recv_message['specs']['POWER']['voltage']/1000
            
            time_of_stage = recv_message['time']
            # [stage, cpu_freq, cpu_temp, gpu_freq, gpu_temp, time, proposal]
            # t2ddl = self.deadline - time_of_stage
            # observations = [stage,
            #                 self.metadata['freq2id']['cpu_freq'][cpu_freq], cpu_temp,
            #                 self.metadata['freq2id']['gpu_freq'][gpu_freq], gpu_temp,]
            cpu_thres = self._throttling_bound - cpu_temp
            gpu_thres = self._throttling_bound - gpu_temp
            observations = [stage,
                            cpu_freq, np.tanh(cpu_thres) if cpu_thres > 0 else cpu_thres,
                            gpu_freq, np.tanh(gpu_thres) if gpu_thres > 0 else gpu_thres,]
            # observations = [stage,
            #                 cpu_freq, cpu_temp,
            #                 gpu_freq, gpu_temp,]
            self.time_of_stage = time_of_stage
            
            self.cooldown = recv_message.get('cooldown')
            self.finish_cd = recv_message.get('finish_cd')
            
            # self.cpu_freq = self.metadata['freq2id']['cpu_freq'][cpu_freq]
            self.cpu_freq = cpu_freq
            # self.gpu_freq = self.metadata['freq2id']['gpu_freq'][gpu_freq]
            self.gpu_freq = gpu_freq
            self.cpu_temp = cpu_temp
            self.gpu_temp = gpu_temp
            self.power = power
            self.stage = stage
            # obs = [stage, cpu_freq, cpu_temp, gpu_freq, gpu_temp, t2ddl, num_proposals]

            if stage == 1:
                self._step_time = time_of_stage
                num_proposals = recv_message['proposals']
                # observations.append(self.deadline - self._step_time)
                self.t2ddl = self.deadline * self.deadline_split - time_of_stage
                observations.append(self.t2ddl)
                observations.append(num_proposals)
                self.stage1_buffer.append(self._step_time)

            elif stage == 2:
                self._step_time += time_of_stage
                self.latency = self._step_time
                # observations.append(self.deadline - self._step_time)
                self.t2ddl = self.deadline - self._step_time
                observations.append(self.t2ddl)
                if not self.slimmable:
                    observations.append(0)  # 0 proposals indicate none if not SlimmableDQN
                self._num_steps += 1
                self.sync_t = recv_message['t']

                self._time_list.append(self._step_time)
                self.stage2_buffer.append(self._step_time)

            else:
                NotImplementedError(f"{stage} should be 0/1/2")
            
            return observations
        else:
            NotImplementedError(f"Type: {recv_message} not supported")
            
    
    @property
    def obs(self):
        # obs = [stage, cpu_freq, cpu_temp, gpu_freq, gpu_temp, t2ddl/total_time, num_proposals]
        obs = torch.tensor(self._get_obs(), dtype=torch.float).cuda() if torch.cuda.is_available() else torch.tensor(obs, dtype=torch.float)
        return obs
        

    def seed(self, seed=None):
        if seed is not None:
            self.np_random, seed_ = seeding.np_random(seed=seed)
            torch.manual_seed(seed)
            return [seed_]

    def reset(self):
        self._rewards = [[], [], [], []]
        self._accumulated_reward = 0
        self.throttling_done = False
        self._num_steps = 0
        self._time_list = []
        if self.random_throttling:
            self._throttling_bound = random.randint(80, int(self._throttling))
            self.socket_server.logger.info(f"Reset throttling to {self._throttling_bound}")

        if hasattr(self, 'pbar'):
            self.pbar.close()
        self.pbar = tqdm(total=self._max_steps, leave=False, dynamic_ncols=True,
                         bar_format='{desc}{percentage:3.0f}%',
                         desc=f"t:{int(self._throttling_bound):d}|R:--||Re[C:-- G:-- D:--]||Info[C:-- G:-- D:--]")
        
        self.previous_cpu_thres = 0
        self.previous_gpu_thres = 0

    def init(self, seed=None):
        super().reset(seed=seed)
        # init_state = [0, 0, 0, 0, 0] # need to change this
        init_state = self.obs
        assert init_state.squeeze()[0] == 2
        if self.slimmable:
            stage, state = torch.split(init_state, [1, init_state.size(0) - 1])
        else:
            stage, state = init_state[0], init_state
        self.reset()
        return int(stage % 2), state.unsqueeze(0)
        
    def random_action(self):
        return self.action_space.sample()
        
    def step(self, action, fan=None):
        action = int(action) if isinstance(action, np.ndarray) else action
        
        Fgpu, Fcpu = self._action_to_frequency[int(action)]
        assert Fcpu in self.metadata['freq']['cpu_freq'] and Fgpu in self.metadata['freq']['gpu_freq']
        
        action = dict(GPU={'freq': Fgpu}, CPU={'freq': Fcpu})
        if fan and isinstance(fan, dict):
            action['FAN'] = fan
        self.socket_server.send(action)
        obs = self.obs
        done = self._ifdone()
        # obs = [stage, cpu_freq, cpu_temp, gpu_freq, gpu_temp, t2ddl, num_proposals]
        reward = self._reward(cpu_temp_thres=obs[2], gpu_temp_thres=obs[4], time2ddl=obs[5])
        self.pbar.set_description_str(f"t:{int(self._throttling_bound):d}|R:{self.reward:.02f}||" \
                                      f"Re[C:{self.rewards[0][-1]:.02f} G:{self.rewards[1][-1]:.02f} D:{self.rewards[2][-1]:.02f}]||"\
                                      f"Info[C:{self.cpu_temp:.02f} G:{self.gpu_temp:.02f} D:{obs[5]:.02f}]")

        if obs[0] == 2:
            obs[5] = self.deadline
            self.pbar.update()            

        # if done:
        #     reward -= self._max_steps - self._num_steps
        return (obs,
                reward,
                done)
        

    def _ifdone(self):
        done = self._num_steps == self._max_steps
        throttling_done = self.cooldown
        if throttling_done:
            while len(self.stage1_buffer) != 1:
                self.stage1_buffer.popleft()
            while len(self.stage2_buffer) != 1:
                self.stage2_buffer.popleft()
        done = done or throttling_done

        self.throttling_done = throttling_done
        self._var, self._mean = torch.var_mean(torch.tensor(self._time_list, dtype=torch.float))
        return done
    
    @property
    def accumulated_reward(self):
        return self._accumulated_reward
    
    @property
    def rewards(self):
        return self._rewards

    def _reward(self, cpu_temp_thres, gpu_temp_thres, time2ddl):
        reward = self._compute_reward(cpu_temp_thres, gpu_temp_thres, time2ddl)
        self._accumulated_reward += reward
        self.reward = reward
        return torch.tensor([reward], dtype=torch.float).cuda()
    
    def _compute_reward(self, cpu_temp_thres, gpu_temp_thres, time2ddl):
        # cpu_temp_reward = F.hardtanh(cpu_temp_thres/5, max_val=1, min_val=0) if cpu_temp_thres > 0 else self.penalty * cpu_temp_thres
        # cpu_temp_reward = torch.tanh(cpu_temp_thres/5) if cpu_temp_thres > 0 else self.penalty * (cpu_temp_thres - 1)
        cpu_temp_reward = cpu_temp_thres if cpu_temp_thres > 0 else self.penalty * cpu_temp_thres

        # cpu_temp_reward = torch.tanh(cpu_temp_thres) if cpu_temp_thres > 0 else self.penalty * cpu_temp_thres
        # gpu_temp_reward = F.hardtanh(gpu_temp_thres/5, max_val=1, min_val=0) if gpu_temp_thres > 0 else self.penalty * gpu_temp_thres
        # gpu_temp_reward = torch.tanh(gpu_temp_thres/5) if gpu_temp_thres > 0 else self.penalty * (gpu_temp_thres - 1)
        gpu_temp_reward = gpu_temp_thres if gpu_temp_thres > 0 else self.penalty * gpu_temp_thres
        
        # gpu_temp_reward = torch.tanh(gpu_temp_thres) if gpu_temp_thres > 0 else self.penalty * gpu_temp_thres

        self.previous_cpu_thres = cpu_temp_thres
        self.previous_gpu_thres = gpu_temp_thres
        
        cpu_temp_reward *= 0.5
        gpu_temp_reward *= 0.5
        temp_reward = cpu_temp_reward + gpu_temp_reward

        # maxmize time2ddl
        # time_reward = torch.tanh(time2ddl/100) if time2ddl > 0 else - self.penalty * torch.ones_like(time2ddl)
        time_reward = F.softsign(time2ddl/1000) if time2ddl > 0 else self.penalty * F.softsign(time2ddl/1000)
        # time_reward = time2ddl/100 if time2ddl > 0 else self.penalty * F.softsign(time2ddl/100)

        time_reward *= self.beta


        self._rewards[0].append(cpu_temp_reward)
        self._rewards[1].append(gpu_temp_reward)
        self._rewards[2].append(time_reward)
        
        if len(getattr(self, f"stage{self.stage}_buffer")) > 0:
            std = torch.std(torch.FloatTensor(getattr(self, f"stage{self.stage}_buffer")), unbiased=False)
            variance_penalty = self.beta / (1 + std)
            time_reward += variance_penalty
            self._rewards[3].append(variance_penalty)

        return temp_reward + time_reward



class TX2Env(BaseEnv):
    
    def __init__(self,
                 max_steps=100,
                 throttling=None,
                 seed=None,
                 deadline=624,
                 penalty=2,
                 beta=10,
                 random_throttling=False,
                 temp_control=False,
                 log_name='logs/gym_server',
                 slimmable=False) -> None:
        super(TX2Env, self).__init__(max_steps=max_steps, throttling=throttling,
                                     seed=seed, deadline=deadline, penalty=penalty,
                                     beta=beta,
                                     random_throttling=random_throttling,
                                     log_name=log_name,
                                     cpu_freq=TX2CPU.FREQ,
                                     gpu_freq=TX2GPU.FREQ,
                                     slimmable=slimmable,
                                     temp_control=temp_control)

    def _init_client(self, throttling=80000):
        spec = dict()
        spec['gov'] = 'userspace'
        # if throttling:
        #     spec['throttling'] = throttling
        sending_message = dict(
            CPU=dict(**spec, freq=self.metadata['freq']['cpu_freq'][0]),
            GPU=dict(**spec, freq=self.metadata['freq']['gpu_freq'][0]),
            FAN=dict(control=0)
        )
        self.socket_server.send(sending_message)

    


class OrinEnv(BaseEnv):    

    def __init__(self,
                 max_steps=100,
                 throttling=None,
                 seed=None,
                 deadline=400,
                 penalty=2,
                 beta=1,
                 random_throttling=False,
                 log_name='logs/orin_server',
                 slimmable=False,
                 deadline_split=1,
                 temp_control=False,
                 fan=0) -> None:
        super(OrinEnv, self).__init__(max_steps=max_steps, throttling=throttling,
                                      seed=seed, deadline=deadline, penalty=penalty,
                                      beta=beta,
                                      random_throttling=random_throttling,
                                      log_name=log_name,
                                      cpu_freq=OrinNanoCPU.FREQ,
                                      gpu_freq=OrinNanoGPU.FREQ,
                                      slimmable=slimmable,
                                      deadline_split=deadline_split,
                                      temp_control=temp_control,
                                      fan=fan)

    

class EMA(object):
    def __init__(self, decay):
        self.decay = decay
        self.data = None

    def update(self, x: torch.Tensor):
        if self.data is None:
            self.data = x.clone()
        else:
            smooth_data = (1.0 - self.decay) * self.data + self.decay * x
            self.data.copy_(smooth_data)
        assert x.shape == self.data.shape, f"{x.shape}, {self.data.shape}"
        return self.data

    def __add__(self, x):
        return self.data + x

    def __sub__(self, x):
        return self.data - x