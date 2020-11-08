import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork_actor(nn.Module):
    def __init__(self, input_dim, out_dim, 
                 hidden_1=512,hidden_2=256, nonlin=F.relu,
                 constrain_out=False, norm_in=True,
                 discrete_action=True):
        super(MLPNetwork_actor, self).__init__()

        hidden_1 = 64
        hidden_2 = 64

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            # self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:
            self.out_fn = lambda x: x

    def forward(self, X):
        h1  = self.nonlin(self.fc1(self.in_fn(X)))
        h2  = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))

        return out


class MLPNetwork_critic(nn.Module):

    def __init__(self, input_dim, out_dim, 
                 hidden_1=1024,hidden_2=512, hidden_3=256,
                 nonlin=F.relu,
                 constrain_out=False, norm_in=True,
                 discrete_action=True):

        super(MLPNetwork_critic, self).__init__()

        hidden_1 = 64
        hidden_2 = 256

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_1)
        self.fc3 = nn.Linear(hidden_1, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:
            self.out_fn = lambda x: x


    def forward(self, X):

        h1  = self.nonlin(self.fc1(self.in_fn(X)))
        h2  = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out