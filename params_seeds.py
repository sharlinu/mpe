import argparse
import os
import sys
import shutil
import json
import time
import datetime

import pdb

def get_params(env_id, seed, n_episodes, episode_length, exp_id, agent_alg):

    print("\n\n\n\n")
    print("==============================")
    print("Acquiring Arguments")
    print("==============================")


    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--buffer_length", default=int(1e6), type=int,
                        help="replay memory buffer capacity")
    parser.add_argument("--n_episodes", default=60000, type=int,
                        help="number of episodes")
    parser.add_argument("--episode_length", default=100, type=int,
                        help="number of steps for episode")
    parser.add_argument("--steps_per_update", default=100, type=int,
                        help="target networks update frequency")
    parser.add_argument("--batch_size", default=1024, type=int,
                        help="batch size for model training")
    parser.add_argument("--n_exploration_eps", default=-1, type=int,
                        help="exploration epsilon, -1: n_episodes")
    parser.add_argument("--init_noise_scale", default=0.3, type=float,
                        help="noise initialization")
    parser.add_argument("--final_noise_scale", default=0.0, type=float,
                        help="noise stop updates value")
    parser.add_argument("--save_epochs", default=5000, type=int,
                        help="save model interval")
    parser.add_argument("--plcy_lr", default=0.0001, type=float,
                        help="learning rate")
    parser.add_argument("--crtc_lr", default=0.001, type=float,
                        help="learning rate")
    parser.add_argument("--tau", default=0.01, type=float,
                        help="soft update parameter")
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--device", default="gpu",
                        choices=['cpu','gpu'], 
                        help="device type")
    parser.add_argument("--plcy_hidden_dim", default=128, type=int, 
                        help="actor hidden state dimension")
    parser.add_argument("--crtc_hidden_dim", default=64, type=int, 
                        help="critic hidden state dimension")
    parser.add_argument("--agent_type", default='MLP',
                        help="agent type")


    # path arguments
    # parser.add_argument('--exp_id', default='maddpg',
    #                     help='experiment ID')
    parser.add_argument('--dir_base', default='./experiments',
                        help='path of the experiment directory')
    parser.add_argument('--port', default=0, type=int,\
                         help='tensorboardX port')
    parser.add_argument('--exp_descr', default='',
                         help='short experiment description')

    # experiment modality
    parser.add_argument('--resume', default='',
                        help='path in case resume is needed')
    parser.add_argument('--expmode', default='normal',
                        help='fast exp mode is usually used to try is code run')
    parser.add_argument("--render", default=0, type=int,
                        help="epochs interval for rendering, 0: no rendering")
    parser.add_argument("--benchmark", action="store_true",
                        help="benchmark mode")
    parser.add_argument("--discrete_action", default="True",
                        choices=['True', 'False'],
                        help="discrete actions")


    # acquire in a dict
    config = parser.parse_args()
    args   = vars(config)

    args['env_id']         = env_id
    args['random_seed']    = seed
    args['n_episodes']     = n_episodes
    args['episode_length'] = episode_length
    args['exp_id']         = '{}seed{}'.format(exp_id,seed)
    args['agent_alg']      = agent_alg


    # set arguments which need dependencies
    dir_exp_name = '{}_{}_{}_physics1'.format(str([datetime.date.today()][0]),
                                  args['env_id'],
                                  #args['agent_type'],
                                  #args['agent_alg'],
                                  args['exp_id'],
                                  #args['random_seed']
                                )

    args['dir_exp'] = '{}/{}'.format(args['dir_base'],dir_exp_name)
    args['dir_summary'] = '{}/summary'.format(args['dir_exp'])
    args['dir_saved_models'] = '{}/saved_models'.format(args['dir_exp'])
    args['dir_monitor'] = '{}/monitor'.format(args['dir_exp'])

    # get current process pid
    args['process_pid'] = os.getpid()

    # creating folders:
    directory = args['dir_exp']
    if os.path.exists(directory) and args['resume'] == '':
        # toDelete= input("{} already exists, delete it if do you want to continue. Delete it? (yes/no) ".\
        #     format(directory))
        toDelete = 'no'

        if toDelete.lower() == 'yes':
            shutil.rmtree(directory)
            print("Directory removed")
        if toDelete == 'No':
            print("It was not possible to continue, an experiment \
                   folder is required.Terminiting here.")
            sys.exit()
    if os.path.exists(directory) == False and args['resume'] == '':
        os.makedirs(directory)
        os.makedirs(args['dir_summary'])
        os.makedirs(args['dir_saved_models'])
        os.makedirs(args['dir_monitor'])
        with open(os.path.expanduser('{}/arguments.txt'.format(args['dir_exp'])), 'w+') as file:
            file.write(json.dumps(args, indent=4, sort_keys=True))
    
    time.sleep(1)

    if args['expmode'] == 'fast':
        args['batch_size'] = 8
        args['max_episode_len'] = 50

    # noise
    if args['n_exploration_eps'] < 0:
        args['n_exploration_eps'] = args['n_episodes']

    # discrete actions
    if args['discrete_action'] == 'True':
        args['discrete_action'] = True
    else:
        args['discrete_action'] = False

    print("\n==> Arguments:")
    for k,v in sorted(args.items()):
        print('{}: {}'.format(k,v))
    print('\n')


    return args
