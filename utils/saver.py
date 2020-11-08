import torch as th
import os
import shutil

class Saver:
    """
        Class for saving and resuming the framework
    """ 

    def __init__(self, args):
        self.args = args
        self.path_ckpt = ''
        self.path_ckpt_ep = ''
        self.path_ckpt_best = ''
        self.path_ckpt_best_avg = ''


    def save_checkpoint(self,
                        save_dict,
                        episode = None,
                        filename = 'ckpt_last.pth.tar', 
                        is_best = None,
                        is_best_avg = None,
                        is_last_periodically = None,
                        save_last = None):
        """
            save on file
        """

        ckpt = self.build_state_ckpt(save_dict, episode)
        path_ckpt = os.path.join(self.args['dir_saved_models'], filename)
        
        # saves the model after every epsidep (time-consuming! it is preferred is_last_periodically instead)
        if save_last is not None: 
            th.save(ckpt, path_ckpt)

        # save last periodically
        if is_last_periodically is not None:
            path_ckpt_lastp = os.path.join(self.args['dir_saved_models'], 'last_periodically.pth.tar')
            th.save(ckpt,path_ckpt_lastp)

        # absolute best
        if is_best is not None:
            path_ckpt_best = os.path.join(self.args['dir_saved_models'], 
                                     'ckpt_best_r{}.pth.tar'.format(is_best))
            th.save(ckpt, path_ckpt_best)
            if os.path.exists(self.path_ckpt_best):
                os.remove(self.path_ckpt_best)
            self.path_ckpt_best = path_ckpt_best
        
        # checkpoint episode to save
        if episode is not None:
            path_ckpt_ep = os.path.join(self.args['dir_saved_models'], 
                            'ckpt_ep{}.pth.tar'.format(episode))
            th.save(ckpt,path_ckpt_ep)
            if os.path.exists(self.path_ckpt_ep):
                os.remove(self.path_ckpt_ep)
            self.path_ckpt_ep = path_ckpt_ep

        # average best
        if is_best_avg is not None:
            path_ckpt_best_avg = os.path.join(self.args['dir_saved_models'], 
                                     'ckpt_best_avg_r{}.pth.tar'.format(is_best_avg))
            th.save(ckpt, path_ckpt_best_avg)          
            if os.path.exists(self.path_ckpt_best_avg):
                os.remove(self.path_ckpt_best_avg)
            self.path_ckpt_best_avg = path_ckpt_best_avg
        

    def build_state_ckpt(self, save_dict, episode):
        """
            build a proper structure with all the info for resuming
        """
        ckpt = ({
            'args'       : self.args,
            'episode': episode,
            'save_dict'  : save_dict
            })

        return ckpt


    def resume_ckpt(self, resume_path=''):
        """
            build a proper structure with all the info for resuming
        """
        if resume_path =='':     
            ckpt = th.load(self.args['resume'])
        else:
            ckpt = th.load(resume_path)    

        self.args = ckpt['args']
        save_dict = ckpt['save_dict']
        episode   = ckpt['episode']

        return self.args, episode, save_dict