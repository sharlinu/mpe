import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import os
import threading
import pickle as pickle
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboardX import SummaryWriter
import torch 
import numpy as np

import pdb

class Summarizer:
    """
        Class fro saving the experiment log files
    """ 
    def __init__(self, path_summary, port, resume=''):

        if resume == '':
            self.__init__from_config(path_summary,port)
        else:
            self.__init__from_file(path_summary,port, resume)


    def __init__from_config(self, path_summary, port):
        self.path_summary = path_summary
        self.writer = SummaryWriter(self.path_summary)
        self.port = port
        self.list_rwd = []
        self.list_pkl = []

        if self.port != 0:
            t = threading.Thread(target=self.launchTensorBoard, args=([]))
            t.start()
 

    def __init__from_file(self, path_summary, port, resume):
        
        p = Path(resume).parents[1]
        print('./{}/summary/log_record.pickle'.format(Path(resume).parents[1]))
        self.path_summary = '{}/summary/'.format(p)
        self.writer = SummaryWriter(self.path_summary)
        self.port = port

        # pdb.set_trace()
        print(path_summary)
        with open('{}/summary/log_record.pickle'.format(p),'rb') as f:
            pckl = pickle.load(f)
            self.list_rwd = [x['reward_total'] for x in pckl]
            self.list_pkl = [x for x in pckl]

        if self.port != 0:
            t = threading.Thread(target=self.launchTensorBoard, args=([]))
            t.start()

    def update_log(self,
        idx_episode, 
        reward_total,
        reward_agents,
        critic_loss = None,
        actor_loss = None,
        to_save=True,
        to_save_plot = 10,
        ):
            
        self.writer.add_scalar('reward_total',reward_total,idx_episode)

        for a in range(len(reward_agents)):
            self.writer.add_scalar('reward_agent_{}'.format(a)
                ,reward_agents[a],idx_episode)
            if critic_loss != None:
                self.writer.add_scalar('critic_loss_{}'.format(a)
                                       ,critic_loss[a],idx_episode)
            if actor_loss != None:
                self.writer.add_scalar('actor_loss_{}'.format(a)
                    ,actor_loss[a],idx_episode)

        # save raw values on file
        self.list_rwd.append(reward_total)
        with open('{}/reward_total.txt'.format(self.path_summary), 'w') as fp:
            for el in self.list_rwd:
                fp.write("{}\n".format(round(el, 2)))

        # save in pickle format
        dct = {
            'idx_episode'  : idx_episode,
            'reward_total' : reward_total,
            'reward_agents': reward_agents,
            'critic_loss'  : critic_loss,
            'actor_loss'   : actor_loss
        }
        self.list_pkl.append(dct)


        # save things on disk
        if to_save:
            # self.save_fig(idx_episode, self.list_rwd)
            self.writer.export_scalars_to_json(
                '{}/summary.json'.format(self.path_summary))
            with open('{}/log_record.pickle'.format(self.path_summary), 'wb') as fp:
                pickle.dump(self.list_pkl, fp)

        if idx_episode % to_save_plot==0:
            self.plot_fig(self.list_rwd, 'reward_total')

    def plot_fig(self, record, name):
        durations_t = torch.FloatTensor(np.asarray(record))

        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(20,15))
        ax.grid(True)
        ax.set_ylabel('Duration')
        ax.set_xlabel('Episode')
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        # plt.yticks(np.arange(-200, 10, 10.0))

        ax.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            ax.plot(means.numpy())

        plt.draw()
        # plt.ylim([-200,10])
        
        fig.savefig('{}/{}.png'.format(self.path_summary,name))
        plt.close(fig)


    def save_fig(self, idx_episode, list_rwd):
        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(20,15))
        ax.plot(range(idx_episode+1), list_rwd)
        ax.grid(True)
        ax.set_ylabel('Total reward')
        ax.set_xlabel('Episode')
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        fig.savefig('{}/reward_total.png'.format(self.path_summary))
        plt.draw()
        plt.close(fig)

    def close(self):
        self.writer.close()

    def launchTensorBoard(self):    
        os.system("tensorboard --logdir={} --port={}".format(
                                            self.path_summary,
                                            self.port))
        return
