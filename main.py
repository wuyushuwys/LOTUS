import argparse
import random
import math
import pickle
import sys
import os
import time
import numpy as np

from typing import Union
from datetime import datetime
from tqdm import tqdm
from itertools import count

import torch
from torch.utils.tensorboard import SummaryWriter

from device_toolkit import set_logging
from envs import TX2Env, OrinEnv
from remote_runner import RemoteRunner
from utils import AverageMeter, CSVWriter, ActionScheduler
from dql import DQL

from pprint import pformat


def main(env: Union[TX2Env, OrinEnv], args):

    # Initiate Logging
    writer = SummaryWriter(log_dir=args.log_dir)
    csv_writer = CSVWriter(os.path.join(args.log_dir, "results.csv"),
                           header=["timestamp", "latency", "gpu_temp", "cpu_temp", "power"])
    latency_meter = AverageMeter()

    num_episodes = args.num_episodes
    engine_kwargs = dict(env=env, args=args, writer=writer)
    engine = DQL(**engine_kwargs)

    n_iterations = 0

    # Initialize the environment and get it's state
    stage, state = env.init()

    if args.fan_actions:
        action_scheduler = ActionScheduler(milestones=[int(v * 2) for v in args.milestones ], actions=args.fan_actions, unit='step')
        logger.info("Using action scheduler")
    
    pbar = tqdm(range(1, num_episodes+1), dynamic_ncols=True)
    for i_episode in pbar:
        latency_meter.reset()
        env.socket_server.logger.info(f"Eposode {i_episode}")
        for t in count():
            writer.add_scalar("CPU/freq", env.cpu_freq, n_iterations)
            writer.add_scalar("CPU/temp", env.cpu_temp, n_iterations)
            writer.add_scalar("GPU/freq", env.gpu_freq, n_iterations)
            writer.add_scalar("GPU/temp", env.gpu_temp, n_iterations)

            action = engine.select_action(state, stage)
            next_state, reward, done = env.step(action, fan=dict(speed=action_scheduler.step()) if args.fan_actions else None)

            if args.slimmable:
                next_stage, next_state = torch.split(next_state, [1, next_state.size(0) - 1])
            else:
                next_stage = next_state[0]
            # Store the transition in memory

            # memory[stage].push(state=state, action=action, next_state=next_state if not done else None, reward=reward)
            if args.slimmable:
                num_states = [5, 6]
                if state.size(-1) == num_states[stage]:
                    engine.memory[stage].push(state=state, action=action, next_state=next_state, reward=reward)
                else:
                    print(f"Expected states size {num_states[stage]} but got {state.size(-1)} for {state} at state {stage}")
            else:
                engine.memory.push(state=state, action=action, next_state=next_state, reward=reward)
            
            writer.add_scalar("Reward/step", reward, n_iterations)
            
            # Perform one step of the optimization
            engine.optimize_model(stage, args.validate)

            # if env.throttling_done and stage == 1:
            # if done and stage == 1 and args.cooldown:
            if env.sync_t and args.cooldown_interavl and env.sync_t % args.cooldown_interavl == 0 and stage == 1 and args.cooldown:
                # print(done, env.sync_t, args.cooldown_interavl, env.sync_t % args.cooldown_interavl)
                env.socket_server.logger.info("Cooldown at device")
                next_state, _, _ = env.step(action, fan=dict(speed=args.fan))
                assert env.finish_cd, "Cooldown Failed"
                if args.slimmable:
                    _, next_state = torch.split(next_state, [1, next_state.size(0) - 1])
                env.socket_server.logger.info("Finish Cooldown")

            # record stage - latency
            writer.add_scalar(f"Time/stage{stage}", env.time_of_stage, n_iterations // 2)
            # Move to the next state
            stage, state = int(next_stage % 2), next_state.unsqueeze(0)
            n_iterations += 1


            if next_stage == 1:
                latency = env.latency
                latency_meter.update(latency, 1)
                writer.add_scalar("Time/Latency", latency, n_iterations // 2)
                csv_writer.write([time.monotonic(), env.latency, env.gpu_temp, env.cpu_temp, env.power])
                
            if done:
                writer.add_scalar("step/total", env._num_steps, i_episode)
                accumulated_reward = env.accumulated_reward.item()
                cpu_temp_reward = sum(env.rewards[0])
                gpu_temp_reward = sum(env.rewards[1])
                time_reward = sum(env.rewards[2])
                variance_penalty = sum(env.rewards[3])
                env.reset()
                break
        if i_episode % args.save_interval == 0: 
            engine.save_checkpoint(args.log_dir)


        writer.add_scalar("Reward/Total", accumulated_reward, i_episode)
        writer.add_scalar("Reward/CPU", cpu_temp_reward, i_episode)
        writer.add_scalar("Reward/GPU", gpu_temp_reward, i_episode)
        if not torch.isnan(time_reward):
            writer.add_scalar("Reward/time", time_reward, i_episode)
        if not torch.isnan(variance_penalty):
            writer.add_scalar("Reward/variance", variance_penalty, i_episode)
        writer.add_scalar("Time/Average_Latency", latency_meter.avg, i_episode)

        env.socket_server.logger.info(f"Eposode {i_episode}:Reward[total, cpu, gpu, time, var]\t[{accumulated_reward:.4f}, {cpu_temp_reward:.4f}, {gpu_temp_reward:.4f}, {time_reward:.4f}, {variance_penalty:.4f}]")

        logger.info(f"Eposode {i_episode}:Reward[total, cpu, gpu, time, var]\t[{accumulated_reward:.4f}, {cpu_temp_reward:.4f}, {gpu_temp_reward:.4f}, {time_reward:.4f}, {variance_penalty:.4f}]")

        pbar.set_description(f"Accumulated Reward: {accumulated_reward:.02f}")

        engine.save_checkpoint(args.log_dir)

        if not args.validate:
            engine.scheduler_step()

    data_path = os.path.join(args.log_dir, "replay_memory.pkl")
    with open(data_path, 'wb') as f:
        pickle.dump(engine.memory, f)
    print(f"Save Replay Memory to {data_path}\tFile size: {os.path.getsize(data_path) / 1024 ** 2:.04f} mb")
    logger.info(f"Save Replay Memory to {data_path}\tFile size: {os.path.getsize(data_path) / 1024 ** 2:.04f} mb")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thermal DRL')
    parser.add_argument('--arch', '-a', required=True, type=str, default='faster_rcnn', choices=['faster_rcnn', 'mask_rcnn'], help='arch for two stage model')
    parser.add_argument('--dataset', '-data', required=True, type=str, default='edgeperf', choices=['edgeperf', 'visdrone2019'], help='evaluation dataset')
    parser.add_argument('--deadline', required=True, type=float, default=624, help='the latency deadline for each frame in ms')
    parser.add_argument('--throttling', required=True, type=int, default=94, help='throttling temperature for device in Celsius (negative for random from [70, 90])')
    parser.add_argument('--random-throttling', action='store_true', help="random throttling for each episodes")
    parser.add_argument('--slimmable', action='store_true', help="slimmable DQN model")
    parser.add_argument('--beta', type=float, default=1, help='beta for latency reward')
    parser.add_argument('--seed', type=int,default=1234)
    parser.add_argument('--buffer-size', type=int,default=4000)
    parser.add_argument('--batch-size', type=int,default=16)
    parser.add_argument('--num-episodes', type=int, default=2000)
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--penalty', type=float, default=2)
    parser.add_argument('--eps-start', type=float, default=0.9, help='the max eps for selecting actions')
    parser.add_argument('--eps-end', type=float, default=0.01, help='the min eps for selecting actions')
    parser.add_argument('--eps-decay', type=float, default=10000, help='the eps for selecting actions')
    parser.add_argument('--tau', type=float, default=0.005, help='TAU')
    parser.add_argument('--gamma', type=float, default=0.99, help='GAMMA')
    parser.add_argument('--save-interval', type=float, default=100, help="save checkpoint interval")
    parser.add_argument('--pretrained', type=str, default=None, help="weight of pretrained model")
    parser.add_argument('--validate', action='store_true', help="validate trained model")
    parser.add_argument('--name', type=str, default=None, help="name for log dir")
    parser.add_argument('--memory', type=str, default=None, help="Load memory for training")
    parser.add_argument('--deadline-split', '-dsp', type=float, default=1, help="deadline split for stage 1")
    parser.add_argument('--fan', '-f', type=int, default=0, help="fan speed")
    parser.add_argument('--temp-control', action='store_true', help="enable temperature fan control")
    parser.add_argument('--yes', '-y', action='store_true', help="Select Yes option")
    # Fan actions
    parser.add_argument('--fan-actions', nargs='+', type=int, default=None, help="fan actions")
    parser.add_argument('--milestones', nargs='+', type=int, default=None, help="fan actions milestones")
    # remote command
    # parser.add_argument('--cooldown', '-cd', type=int, default=None, help="cooldown interval on device")
    parser.add_argument('--cooldown', '-cd', action='store_true', help="cooldown training mode")
    parser.add_argument('--cooldown-interavl', '-cdi', type=int, default=None, help="time for each cooldown (random when negetive)")
    # debug
    parser.add_argument('--debug', action='store_true', help="debug mode")

    args = parser.parse_args()

    name = '_'.join([args.name, datetime.now().strftime('%Y_%m_%d_%H%M%S')]) if args.name else datetime.now().strftime("%Y_%m_%d_%H_%M")
    
    if args.cooldown:
        prefix = f"cd_"
        if args.cooldown_interavl:
            prefix += f"{args.cooldown_interavl}_"
        name = prefix + name
    
    name = '_'.join([args.arch, f"beta_{args.beta}", args.dataset, f"t_{args.throttling}", f"f_{args.fan}", f"ddl_{int(args.deadline)}",name])
    
    print(f"Runing {name}")
    
    args.log_dir = os.path.join("logs", "train" if not args.validate else "validate", name)
    
    if args.validate:
        args.num_episodes = 400
    
    if os.path.exists(args.log_dir):
        while True:
            response = input(f"{args.log_dir} already exists. Do you want to overwirte it? [y/n]") if not args.yes else 'y'
            if response == 'y':
                break
            elif response == 'n':
                sys.exit()
    else:
        os.makedirs(args.log_dir, exist_ok=True)

    RemoteRunner(cmd=f"python ~/{args.arch}/test_fan.py 50", logging_path=os.path.join(args.log_dir, 'remote_fan'), std_out=True).process.wait()

    command = f"sudo -E python ~/{args.arch}/inference.py --dataset {args.dataset}"

    if args.cooldown and not args.validate:
        if args.cooldown_interavl:
            command += f" -cd -i {args.cooldown_interavl} --fan {args.fan}"
        else:
            command += f" -cd -i {args.max_steps} --fan {args.fan} --throttling {args.throttling}"

    logger = set_logging(name=os.path.join(args.log_dir, 'log'))
    print(args.log_dir)
    logger.info(pformat(args))
    logger.info(f"Execute command: {command}")
    remote_runner = RemoteRunner(cmd=command, logging_path=os.path.join(args.log_dir, 'remote_script'))

    # args.eps_decay = (args.num_episodes * args.max_steps * 2) // 10

    throttling = args.throttling if args.throttling > 0 else random.randint(75, 90)
    env = OrinEnv(max_steps=args.max_steps,
                  throttling=int(throttling * 1000),
                  seed=args.seed,
                  deadline=args.deadline,
                  penalty=args.penalty,
                  beta=args.beta,
                  random_throttling=args.random_throttling,
                  log_name=os.path.join(args.log_dir, 'envs'),
                  slimmable=args.slimmable,
                  deadline_split=args.deadline_split,
                  temp_control=args.temp_control,
                  fan=args.fan)
    
    try:
        main(env, args)
    except KeyboardInterrupt:
        env.socket_server.close()
        logger.info('End server side')
    remote_runner.process.kill()
    logger.warning("Kill remote progress")
    logger.warning("Done")