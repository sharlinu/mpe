import numpy as np
from multiagentsha.core import World, Agent, Landmark
from multiagentsha.scenario import BaseScenario
import pdb
from scipy.spatial import distance

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 3
        num_agents = 3
        num_landmarks = 3
        max_vision = 0.6
        value_outvision = 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.max_vision = max_vision
            agent.value_outvision = value_outvision
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        occupying_agents = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            for i in range(len(dists)):
                if dists[i] >= agent.max_vision:
                    dists[i] = agent.value_outvision
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.2:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    #rew -= 1
                    collisions += 1
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in world.landmarks]
                for i in range(len(dists)):
                    if dists[i] >= agent.max_vision:
                        dists[i] = agent.value_outvision
                if min(dists) < 0.2:
                    occupying_agents += 1                                
        return (rew, collisions, min_dists, occupied_landmarks, occupying_agents)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            for i in range(len(dists)):
                if dists[i] >= agent.max_vision:
                    dists[i] = agent.value_outvision
            rew -= min(dists)
            # pdb.set_trace()
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            rpos = entity.state.p_pos - agent.state.p_pos
            d = distance.euclidean(entity.state.p_pos,agent.state.p_pos)
            if np.abs(d) >= agent.max_vision:
                for i in range(rpos.shape[0]):
                    rpos[i] = agent.value_outvision
            entity_pos.append(rpos)

            # entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            rpos = other.state.p_pos - agent.state.p_pos
            d = distance.euclidean(other.state.p_pos,agent.state.p_pos)
            if np.abs(d) >= agent.max_vision:
                for i in range(rpos.shape[0]):
                    rpos[i] = agent.value_outvision
            other_pos.append(rpos)

            # other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
