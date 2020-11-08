from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from utils.noise import OUNoise, MyNoise
from .networks import *
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits


class DDPGAgent_MLP(object):
    """
    Class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, 
                 plcy_hidden_dim=64, crtc_hidden_dim=64, 
                 plcy_lr=0.01, crtc_lr=0.01,
                 discrete_action=True):
        self.policy = MLPNetwork_actor(num_in_pol, num_out_pol,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.target_policy = MLPNetwork_actor(num_in_pol, num_out_pol,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.critic = MLPNetwork_critic(num_in_critic, 1,
                                 constrain_out=False,
                                 discrete_action=discrete_action)
        self.target_critic = MLPNetwork_critic(num_in_critic, 1,
                                discrete_action=discrete_action)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=plcy_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=crtc_lr)
        
        if discrete_action:
            self.exploration = 0.3
        else:
            self.exploration = OUNoise(num_out_pol)

        self.discrete_action = discrete_action


    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()
        return


    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale


    def step(self, obs, explore=False):
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action


    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}


    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])