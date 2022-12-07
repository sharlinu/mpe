"""
    Example file to interact with environments
"""
import numpy as np
import time
from make_env import make_env

# inizialize the environment object
env  = make_env('simple_box')

# get some env properties
n_agents          = env.n                 # number of agents
observation_space = env.observation_space # observation space
action_space      = env.action_space      # action space

# print them to see their format
print("n_agents: {}".format(n_agents))
print("observation_space: {}".format(observation_space))
print("action_space: {}".format(action_space))


# This is the scheleton of a training process
# Each episode is composed by a specified number of time-steps (which
# so represent the length of each episode)
# At each time-step 
# - agents access to their current observations
# - chose the actions (in this case random) to do
# - send these actions to the environment which in return will send
# the current reward and the next observation

n_episodes  = 3
n_timesteps = 10

for e in range(n_episodes): # episodes start here
    # resetting the environement will give a new starting state
    # (we do this at the beginning of each episode)
    observation = env.reset()
    print('initial observation', observation)
    for t in range(n_timesteps):
        # this is to render and it's completely optional
        # is not needed during the training
        env.render()

        # now we are going to take the actions for each agent
        agent_actions = [] # list containing the actions of all agents
        for i_agent in range(n_agents):

            # here i_agent will produce a random action
            # (in the real system this would done by our algo) 
            agent_action_space = env.action_space[i_agent]
            action             = agent_action_space.sample()    
            action_vec         = np.zeros(agent_action_space.n)
            action_vec[action] = 1 # TODO see how that changes
            agent_actions.append(action_vec)

        print('-------timestep', t)
        # At this point we send the actions of all the agent to the
        # environment and in return he will return us:
        # observation: list of next observations (an observation for each agent)
        # reward: list of current reward (a reward for each agent)
        # done: list flags (1: episode finished, 0 episode not finished),
        # info: additional infos (can be optional)
        observation, reward, done, info = env.step(agent_actions)
        print('--- new details are:')
        print('p_pos', env.world.agents[0].state.p_pos)
        print('p_vel', env.world.agents[0].state.p_vel)
        print('action', env.world.agents[0].action.u)
        print('reward', reward)
        print('done', done)
        print('observation', observation)
        time.sleep(0.5)
        if any(done):
            print('episode finsihed')
            break
