import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from random import shuffle

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        self.num_agents = 3
        self.num_landmarks = self.num_agents
        # generate random colors
        rcolor = lambda : list(np.random.uniform(0, 1, 3))
        self.colors = [rcolor() for _ in range(self.num_agents)]
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.id = i
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.id = i
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents

        # Landmark - Agent random assignment 
        randi = list(range(self.num_agents))
        shuffle(randi)
        for i in range(len(randi)):
            world.landmarks[i].color = self.colors[i]
            world.agents[i].color    = self.colors[i]
            world.agents[i].target   = world.landmarks[i].id

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1.7, +1.7, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c     = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1.7, +1.7, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # calculate distange between agent and target
        agent_pos        = agent.state.p_pos
        agent_target_pos = world.landmarks[agent.target].state.p_pos
        tdistance        = np.sqrt(np.sum(np.square(agent_pos - agent_target_pos)))
        # if target is reached assign high reward
        rew              = 0
        collisions       = 0
        landmark_reached = 0
        landmark_pos     = []
        if tdistance < agent.size*2:
            landmark_reached = 1
            rew += 10
        else:
            rew -= tdistance # add negative distance to maximize it later
        # penalize in case of collisions
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 50 
                    collisions += 1
        for i, landmark in enumerate(world.landmarks):
            landmark_pos.append(landmark.state.p_pos.tolist())
        return (rew, collisions, landmark_reached, landmark_pos)

    def reward(self, agent, world):
        # REWARD IS NOT SHARED HERE, NEEDS TO BE SHARED LATER IN THE CODE

        # calculate distange between agent and target
        agent_pos        = agent.state.p_pos
        agent_target_pos = world.landmarks[agent.target].state.p_pos
        tdistance        = np.sqrt(np.sum(np.square(agent_pos - agent_target_pos)))
        # if target is reached assign high reward
        rew = 0
        if tdistance < agent.size*2:
            rew += 10
        else:
            rew -= tdistance # add negative distance to maximize it later
        # penalize in case of collisions
        if agent.collide:
            for a in world.agents:
                if a.id != agent.id:
                    if self.is_collision(a, agent):
                        rew -= 50
        return rew

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            if entity.id == agent.target:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel]+entity_pos+other_pos)
