import numpy as np
from multiagentsha.core import World, Agent, Landmark
from multiagentsha.scenario import BaseScenario


colors = {
    1: [0.37, 1, 0.5],  # gem color
    0: [0.25,0.25,0.25], # agent color
    80: [247, 200, 124] # agent color when they pick up something
}


class Scenario(BaseScenario):
    def _manhattan(self, a, b):
        return sum(abs(val1 - val2) for val1, val2 in zip(a, b))

    def make_world(self):
        world = World()
        #world.dt = 1
        #world.damping = 1
        # set any world properties first
        world.dim_c = 0
        num_agents = 2
        num_landmarks = 2
        goal_length = 1
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        #world.landmark_colors = np.array(n_colors = goal_length)
        world.landmark_colors = np.arange(goal_length)

        for i, agent in enumerate(world.agents):
            #agent.accel = 1
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.ghost = False # this would mean that agent can pass through walls
            agent.holding = None

            agent.size = 0.15 # TODO change sizes
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            # landmark.color = world.landmark_colors
            landmark.alive = True
            landmark.collide = False
            landmark.movable = False
            # landmark.boundary would make sure that landmarks are not placed on the boundary
        # make initial conditions

        self.reset_world(world)
        return world

    def post_step(self, world):
        #print('processing post step')
        # self.reset_cached
        for l in world.landmarks:
            if l.alive:
                for a in world.agents:
                    if a.holding is None and self.is_collision(l,a):
                        l.alive = False
                        a.holding = True
                        a.color = l.color
                        #print(f'take over {l.color} from {a.color}')
                        l.state.p_pos = np.array([-999.,-999.]) # TODO check how that shows up on observations or why -999
            # in treasure_collection we would have to respawn the treasures but that is not needed here as they do not come back
        # treasure_collection has deposit, which could be here equivalent to our locks that need to be moved to
    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array(colors[0])
            agent.holding = None
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array(colors[1])# TODO generalise
        # set random initial states
        for agent in world.agents:
            #agent.state.p_pos = np.ones(world.dim_p)*0.5
            agent.state.p_pos = np.random.uniform(low=-1, high=+1, size=world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.alive = True
        # TODO in treasure we have calculate_distance?

    def benchmark_data(self, agent, world):
        global dist, l_pos
        rew = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [self._manhattan(a.state.p_pos, l.state.p_pos) for a in world.agents]
            #dists = [np.sum(np.square(a.state.p_pos - l.state.p_pos)) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
        # TODO change more to treasure collection
        collected_keys = 0
        for l in world.landmarks:
            dist = float(self._manhattan(l.state.p_pos, agent.state.p_pos))
            #dist = np.sum(np.square(agent.state.p_pos - l.state.p_pos))
            l_pos = l.state.p_pos.tolist()
            if dist == 0:
                collected_keys += 1
        return (agent.state.p_pos.tolist(), agent.action.u.tolist(), l_pos, dist, collected_keys)

    def is_collision(self, agent1, agent2):
        dist = self._manhattan(agent1.state.p_pos, agent2.state.p_pos)
        #dist = np.sum(np.square(agent1.state.p_pos - agent2.state.p_pos))
        return True if dist < 0.2 else False

    def reward(self, agent, world):
        rew = 0
        # reward for collecting keys
        for t in world.landmarks:
            rew += sum(self.is_collision(a,t) for a in world.agents if a.holding is None) * 5 # TODO scale i.e. *5
        # TODO add here deposit when we generalise to MA and receive collection reward
        for l in world.landmarks:
            # TODO this would just be a simplification but we do not want to have dense rewards?
            dists = [self._manhattan(a.state.p_pos, l.state.p_pos) for a in world.agents]
            #dists = [np.sum(np.square(a.state.p_pos - l.state.p_pos)) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            pass
            # TODO do we want penalty for colliding agents?
#            for a in world.agents:
#                if self.is_collision(a, agent):
#                    rew -= 1
        return rew

    def done(self, agent, world):
        #print('testing for done')
        remaining_landmarks = any([l.alive for l in world.landmarks])
        #print('remaining landmarks', remaining_landmarks)
        if remaining_landmarks:
            return False
        else:
            #print('no landmarks left')
            return True

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agentscol
        #comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            #comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # TODO should we add color of other entities to observation?
        # TODO check if treasure collection actually appends boolean vectors to observation
        #print(agent.state.p_pos, agent.color, entity_pos, other_pos)
        out = np.concatenate([agent.state.p_pos] + [agent.color] + entity_pos + other_pos)
        return out
        # + comm

# TODO reward shaping?
# TODO walls to stop agent going off on the sides
