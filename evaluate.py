import argparse
import torch
import time
import imageio
import os
import utils.saver as Saver
import numpy as np
import scipy
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
import json


def run(config):

    saver = Saver.Saver(config)

    config_path, episode, save_dict = saver.resume_ckpt(config['ckpt_path'])
    maddpg = MADDPG.init_from_dict(save_dict)

    # create folder for evaluating
    eval_path = '{}/evaluate'.format(Path(config['ckpt_path']).parents[1])
    print('Saving evaluation at {}'.format(eval_path))
    os.makedirs(eval_path, exist_ok=True)

    gif_path = '{}/{}'.format(eval_path, 'gifs')
    os.makedirs(gif_path, exist_ok=True)

    # if config['save_gifs']:
    #     gif_path = '{}/{}'.format(eval_path, 'gifs')

    # set enviroment
    if config['env_id'] == 'waterworld':
        env = MAWaterWorld_mod(n_pursuers=2, 
                               n_evaders=50,
                               n_poison=50, 
                               obstacle_radius=0.04,
                               food_reward=10.,
                               poison_reward=-1.,
                               ncounter_reward=0.01,
                               n_coop=2,
                               sensor_range=0.2, 
                               obstacle_loc=None, )
        # just for compatibility
        env.n = env.n_pursuers
        env.observation_space = []
        env.action_space = []
        for i in range(env.n):
            env.observation_space.append(np.zeros(213))
            env.action_space.append(2)
    else:
        env = make_env(config['env_id'], 
                      benchmark=config['benchmark'],
                      discrete_action=maddpg.discrete_action)

    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config['fps']  # inter-frame interval

    collect_data = {}


    for ep_i in range(config['n_episodes']):
        print("Episode %i of %i" % (ep_i + 1, config['n_episodes']))
        obs = env.reset()

        frames = []

        collect_item = {
            'ep': ep_i,
            'final_reward': 0,
            'l_infos': [],
            'l_rewards': [],
            'finished': 0,
        }
        l_rewards = []

        for t_i in range(config['episode_length']):
            calc_start = time.time()

            if config['no_render'] != False:
                if config['env_id'] == 'waterworld':
                    frames.append(scipy.misc.imresize(env.render(), (300, 300)))
                else:
                    frames.append(env.render(mode='rgb_array', close=False)[0])   


            # rearrange observations to be per agent, and convert to torch
            # Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            print(torch_actions)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)
            collect_item['final_reward'] = sum(rewards)
            collect_item['l_rewards'].append(sum(rewards))
            collect_item['l_infos'].append(infos)

            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)

            if all(dones):
                collect_item['finished'] = 1
                break
        collect_data[ep_i] = collect_item

        if config['save_gifs']:
            imageio.mimsave('{}/{}.gif'.format(gif_path, ep_i),
                            frames, duration=ifi)

    with open('{}/collected_data.json'.format(eval_path), 'w') as outfile:
        json.dump(collect_data, outfile,indent=4)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("ckpt_path",
                        help="Path of model")
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=30, type=int)
    parser.add_argument("--episode_length", default=30, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--benchmark", action="store_false",
                        help="benchmark mode")
    parser.add_argument("--no_render", action="store_false",
                        help="render")


    config = vars(parser.parse_args())

    run(config)

