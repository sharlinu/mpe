import torch
import torch.nn.functional as F
from gym.spaces import Box
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from agents.DDPGAgent_MLP import DDPGAgent_MLP

import pdb

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG and MADDPG agents
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01,
                 plcy_lr=0.01, crtc_lr=0.01, 
                 plcy_hidden_dim=64, crtc_hidden_dim=64, agent_type='all_1mlp',
                 discrete_action=True):
        self.nagents = len(alg_types)
        self.alg_types = alg_types

        if agent_type == 'MLP':
            Agent = DDPGAgent_MLP
        else: 
            print("Agent type not valid")

        # check parameters here
        self.agents = [Agent(plcy_lr = plcy_lr,
                             crtc_lr = crtc_lr,
                             plcy_hidden_dim = plcy_hidden_dim,
                             crtc_hidden_dim = crtc_hidden_dim,
                             discrete_action = discrete_action,
                             **params)
                       for params in agent_init_params]

        print("Policy network")
        print(self.agents[0].policy)
        print("Critic network")
        print(self.agents[0].critic)

        self.agent_type = agent_type
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau   = tau
        self.plcy_lr = plcy_lr
        self.crtc_lr = crtc_lr
        self.niter = 0
        self.discrete_action = discrete_action


    @property
    def policies(self):
        return [a.policy for a in self.agents]


    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]
        

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)


    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()


    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                 observations)]


    def update(self, sample, agent_i):
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()

        # apply policy and prepare inputs
        if self.alg_types[agent_i] == 'MADDPG':
            if self.discrete_action: # one-hot encode
                all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in zip(self.target_policies,
                                                             next_obs)]
            else:
                all_trgt_acs = [(pi(nobs)) for pi, nobs in zip(self.target_policies,
                                                             next_obs)]               
            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        else:  # DDPG
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                next_obs[agent_i]))),
                                       dim=1)
            trgt_vf_in = torch.cat((next_obs[agent_i],
                                    curr_agent.target_policy(next_obs[agent_i])),
                                   dim=1)
        
        # get vf targets with discount reward
        if self.agent_type == 'a_water_c_water':
            target_value = (rews[agent_i].view(-1, 1) + 
                        self.gamma * curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))
        else:
            target_value = (rews[agent_i].view(-1, 1) + 
                            self.gamma * curr_agent.target_critic(trgt_vf_in) *
                            (1 - dones[agent_i].view(-1, 1)))            

        # prepare inputs
        if self.alg_types[agent_i] == 'MADDPG':
            vf_in = torch.cat((*obs, *acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)

        # get vf
        actual_value = curr_agent.critic(vf_in)

        # mse loss
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        
        # clip and normalize gradients
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5) # TODO change to torch.nn.utils.clip_grad_norm_ and see if that works
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()



        if self.discrete_action:
            curr_pol_out = curr_agent.policy(obs[agent_i]) # here we get out a cuda tensor
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out   = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        if self.alg_types[agent_i] == 'MADDPG':
            all_pol_acs = []
            for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                if i == agent_i:
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.discrete_action:
                    all_pol_acs.append(onehot_from_logits(pi(ob)))
                else:
                    all_pol_acs.append(pi(ob))
            vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                              dim=1)
        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()

        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

        return vf_loss, pol_loss


    def update_all_targets(self):
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1


    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()

        for a in self.agents:
            a.policy = fn(a.policy)
            a.critic = fn(a.critic)
            a.target_policy = fn(a.target_policy)
            a.target_critic = fn(a.target_critic)


    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()

        for a in self.agents:
            a.policy = fn(a.policy)


    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        # move parameters to CPU before saving
        self.prep_training(device='cpu')  
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, str(filename))


    def get_save_dict(self):
        """
        Save trained parameters of all agents into one file
        """
        # move parameters to CPU before saving
        self.prep_training(device='cpu')  
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        return save_dict

    @classmethod
    def init_from_env(cls, env, env_id, 
                      agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, 
                      plcy_lr=0.01, crtc_lr=0.01, 
                      plcy_hidden_dim=64, crtc_hidden_dim=64,
                      agent_type='all_mlp1',
                      discrete_action=True):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        #alg_types = ['agent']
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]

            # here we are assuming discrete actions
            if env_id == 'waterworld':
                get_shape = lambda x: x
            if env_id == 'simple_reference':
                get_shape = lambda x: x.shape
            else:
                get_shape = lambda x: x.n

            num_out_pol = get_shape(acsp)
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)

            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 
                     'plcy_lr': plcy_lr,
                     'crtc_lr': crtc_lr,
                     'plcy_hidden_dim': plcy_hidden_dim,
                     'crtc_hidden_dim': crtc_hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action'  : discrete_action,
                     'agent_type': agent_type}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance


    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(str(filename))
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance


    @classmethod
    def init_from_dict(cls, save_dict):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance