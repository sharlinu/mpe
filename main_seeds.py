import params_seeds
import time
import imageio
import scipy
import numpy as np
import torch as th
import utils.saver as Saver # TODO read through
import utils.summarizer as Summarizer # TODO read through
from torch.autograd import Variable
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import WrapperEnv
from algorithms.maddpg import MADDPG
import pdb # TODO can be removed because not used?

def run(config):
    th.set_num_threads(1)

    if config['resume'] != '':
        resume_path = config['resume']
        n_episodes = config['n_episodes']
        saver = Saver.Saver(config) # TODO read through Saver
        config, start_episode, save_dict = saver.resume_ckpt()
        config['resume'] = resume_path
        config['n_episodes'] = n_episodes


    # seeds
    np.random.seed(config['random_seed'])
    th.manual_seed(config['random_seed'])

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
                      discrete_action=config['discrete_action'])

    env.seed(config['random_seed'])
    env = WrapperEnv(env, config['env_id'])

    # utils
    summary = Summarizer.Summarizer(config['dir_summary'],
                                    config['port'],
                                    config['resume'])

    #hparams= {'dt': env.env.world.dt }
    #summary.writer.add_hparams(hparams)
    saver = Saver.Saver(config)
    
        
    if config['resume'] != '':
        maddpg = MADDPG.init_from_dict(save_dict)
    else:
        start_episode = 0
        maddpg = MADDPG.init_from_env(
                                      env             = env, 
                                      agent_alg       = config['agent_alg'],
                                      adversary_alg   = config['adversary_alg'],
                                      tau             = config['tau'],
                                      plcy_lr         = config['plcy_lr'],
                                      crtc_lr         = config['crtc_lr'],
                                      plcy_hidden_dim = config['plcy_hidden_dim'],
                                      crtc_hidden_dim = config['crtc_hidden_dim'],
                                      agent_type      = config['agent_type'],
                                      env_id          = config['env_id'],
                                      discrete_action = config['discrete_action']
                                      )

    if config['env_id'] == 'waterworld':
        replay_buffer = ReplayBuffer(config['buffer_length'], 
                                 maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp for acsp in env.action_space])
    elif config['env_id'] == 'simple_reference':
        replay_buffer = ReplayBuffer(config['buffer_length'], 
                                     maddpg.nagents,
                                     [obsp.shape[0] for obsp in env.observation_space],
                                     [acsp.shape for acsp in env.action_space])        
    else:        
        replay_buffer = ReplayBuffer(config['buffer_length'],  # by default int(1e6)
                                     maddpg.nagents,
                                     [obsp.shape[0] for obsp in env.observation_space],
                                     [acsp.n for acsp in env.action_space])
    t = 0
    reward_best = float("-inf")
    avg_reward = float("-inf")
    avg_reward_best = float("-inf")
    l_rewards = []
    total_time_start = time.time()

    for ep_i in range(start_episode, config['n_episodes']):

        obs = env.reset()
        maddpg.prep_rollouts(device="cpu")

        # n_exploration_eps = -1 is default and means that all n_episodes have exploration
        explr_pct_remaining = max(0, config['n_exploration_eps'] - ep_i) / config['n_exploration_eps']
        # Slowly reduces the epsilon over time
        maddpg.scale_noise(config['final_noise_scale'] + (config['init_noise_scale'] - config['final_noise_scale']) * explr_pct_remaining)
        maddpg.reset_noise() # TODO what does the reset_noise do here?

        # maddpg.scale_noise(0.999998)


        episode_reward_total = 0.0
        is_best              = False
        is_best_avg          = False
        vf_losses            = None
        pol_losses           = None

        if config['render'] > 0:
            frames = []

        episode_time_start = time.time()

        # TODO this potentially needs to be changed to while not any(done):
        for et_i in range(1,config['episode_length']+1):
            # if we do render, we do so in config['render']-intervals
            if config['render'] > 0 and ep_i % config['render'] == 0:
                if config['env_id'] == 'waterworld':
                    frames.append(scipy.misc.imresize(env.env.render(), (300, 300)))
                else:
                    frames.append(env.env.render(mode='rgb_array', close=False)[0])   

            # rearrange observations to be per agent, and convert to torch Variable
            # TODO check here what this rearranging does? Need to check what observation comes in
            torch_obs = [Variable(th.Tensor(np.vstack(obs[:,i])), # TODO what does torch.autograd.Variable do?
                                  requires_grad=False)
                        for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions
            actions = [[ac[0] for ac in agent_actions]]
            #actions = [np.array([[0,0,0,1,0]], dtype=np.float32)]
            next_obs, rewards, dones, infos = env.step(actions)
            episode_reward_total  += rewards.sum()
            #print('-- observation', obs)
            #print('-- actions', actions)
            #print('-- next observation', next_obs)
            #print('-- reward', rewards)
            #print('-- dones', dones)
            if 'treasure_hunting' in config['env_id']:
                rewards_bff = rewards.copy()
                for i in range(rewards_bff.shape[1]):
                    rewards_bff[0][i] = rewards.mean()
            else:
                rewards_bff = rewards

            # note here that the buffer takes in tensors / Variables for the obs but arrays for actions and rewards ( I think)
            replay_buffer.push(obs, agent_actions, rewards_bff, next_obs, dones)
            obs = next_obs
            t += 1
            if (len(replay_buffer) >= config['batch_size'] and
                (t % config['steps_per_update']) == 0):
                maddpg.prep_training(device=config['device'])

                 # loss for each agent (for log only)
                vf_losses  = []
                pol_losses = []
                
                for a_i in range(maddpg.nagents):
                    if config['device'] == "gpu":
                        sample = replay_buffer.sample(config['batch_size'],
                                                      to_gpu=True)
                    else:
                        sample = replay_buffer.sample(config['batch_size'],
                                                  to_gpu=False)                         
                    vf_loss, pol_loss = maddpg.update(sample, a_i)
                    vf_losses.append(vf_loss)
                    pol_losses.append(pol_loss)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device="cpu")
            #print(dones)
            #print(type(dones))
            if dones.all() == True:
                print('done with time step', et_i)
                break
        if config['env_id'] == 'simple_box':
            ep_rews = replay_buffer.get_average_rewards(et_i)
        else:
            ep_rews = replay_buffer.get_average_rewards(
            config['episode_length'])
        #print('compute average reward with step length ', et_i)


        # check if it was the best model so far
        if episode_reward_total >= reward_best:
            reward_best = episode_reward_total
            is_best = True
        # check if it in average was the best model so far
        l_rewards.append(episode_reward_total)
        th_l_rewards = th.FloatTensor(np.asarray(l_rewards))
        if len(th_l_rewards) >= 100: # TODO check this if statement
            avg_rewards = th_l_rewards.unfold(0, 100, 1).mean(1).view(-1)
            avg_rewards = th.cat((th.zeros(99), avg_rewards))
            avg_reward = avg_rewards[-1]
            if avg_reward > avg_reward_best:
                avg_reward_best = avg_reward
                is_best_avg = True 

        print("==> Episode {} of {}".format(ep_i + 1, config['n_episodes']))
        print('  | Dir exp: {}'.format(config['dir_exp']))
        print('  | Id exp: {}'.format(config['exp_id']))
        print('  | Exp description: {}'.format(config['exp_descr']))
        print('  | Env: {}'.format(config['env_id']))
        print('  | Process pid: {}'.format(config['process_pid']))
        print('  | Tensorboard port: {}'.format(config['port']))
        print('  | Episode total reward: {}'.format(episode_reward_total))
        print('  | Time episode: {}'.format(time.time()-episode_time_start))
        print('  | Time total: {}'.format(time.time()-total_time_start))

        # update logs and save model
        if ep_i % config['save_epochs'] == 0 or ep_i == config['n_episodes']-1:
            ep_save = ep_i+1
        else:
            ep_save = None
        if is_best:
            is_best_save = reward_best
        else:
            is_best_save = None
        if is_best_avg:
            is_best_avg_save = avg_reward_best
        else:
            is_best_avg_save = None
        if all(it is None for it in [ep_save, is_best_save, is_best_avg_save]):
            to_save = False
        else:
            to_save = True
        if to_save:
            saver.save_checkpoint(
                                  save_dict   = maddpg.get_save_dict(),
                                  episode     = ep_save,
                                  is_best     = is_best_save,
                                  is_best_avg = is_best_avg_save
                                  ) 

        summary.update_log(ep_i,
            episode_reward_total,
            ep_rews,
            vf_losses,
            pol_losses,
            to_save = to_save,
            )

        # save gif
        if config['render'] > 0 and ep_i % config['render'] == 0:
            if config['env_id'] == 'waterworld':
                imageio.mimsave('{}/{}.gif'.format(config['dir_monitor'],ep_i),
                        frames[0::3])
            else:
                imageio.mimsave('{}/{}.gif'.format(config['dir_monitor'],ep_i),
                            frames)

    env.close()
    summary.close()

    print("Terminating")

if __name__ == '__main__':
    import glob
    import os
    import shutil

    list_exp_dir = []

    
    agent_alg = 'MADDPG'
    env_id = 'simple_box'


    id_exp = 'std'
    #id_exp = 'try'
    if id_exp == 'try':
        seeds = [1]
        train_n_episodes = 1
        train_episode_length = 20
        test_n_episodes = 5
        test_episode_length = 25
    else:
        seeds = [2001]
        train_n_episodes = 7000
        train_episode_length = 50
        test_n_episodes = 5
        test_episode_length = 50

    dir_collected_data = './experiments/maddpg_multipleseeds_data_{}_{}_{}'.format(agent_alg,env_id,id_exp)
    if os.path.exists(dir_collected_data):
        toDelete= input("{} already exists, delete it if do you want to continue. Delete it? (yes/no) ".\
            format(dir_collected_data))
        if toDelete.lower() == 'yes':
            shutil.rmtree(dir_collected_data)
            print("Directory removed")
            os.makedirs(dir_collected_data)
    else:
        os.makedirs(dir_collected_data)

    for seed in seeds:
        config = params_seeds.get_params(env_id = env_id,
                                        seed = seed,
                                        n_episodes = train_n_episodes,
                                        episode_length = train_episode_length,
                                        exp_id=id_exp,
                                        agent_alg=agent_alg)

        list_exp_dir.append(config['dir_exp'])

        # if seed has been done before
        if  os.path.exists('{}/collected_data_seed_{}.json'.format(dir_collected_data,seed)) == False:
            # train
            run(config)
            
            # test
            model_path = glob.glob('{}/saved_models/ckpt_best_*'.format(config['dir_exp']))[0]
            cmd_test = 'python evaluate.py {} {} --n_episodes {} --episode_length {} --no_render'.format(env_id, model_path, test_n_episodes, test_episode_length)
            print(cmd_test)
            os.system(cmd_test)

            # save files to dir collected data
            shutil.copyfile('{}/evaluate/collected_data.json'.format(list_exp_dir[-1]),
                    '{}/collected_data_seed_{}.json'.format(dir_collected_data,seed)
                    )

            # save files to dir collected data
            shutil.copyfile('{}/summary/reward_total.txt'.format(list_exp_dir[-1]),
                    '{}/reward_training_seed{}.txt'.format(dir_collected_data,seed)
                    )
