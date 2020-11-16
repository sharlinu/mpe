import numpy as np
from multiagentsha.core import World, Agent, Landmark
from multiagentsha.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)

        
        # define config settings here
        config=dict(dt=5.0,
                    self_attend=False,
                    log_scale=ray.tune.grid_search([True]),
                    maximum_distance=np.inf,
                    end_if_alone=True,
                    safe_distance=nm2m(5.0),
                    new_agents=True,
                    new_sector=True,
                    conflict_penalty_coef=ray.tune.grid_search([50.]),
                    conflict_penalty_decay=ray.tune.grid_search([1.]),
                    alert_penalty_coef=ray.tune.grid_search([20.]),
                    alert_penalty_decay=ray.tune.grid_search([1.]),
                    alert_penalty_t_decay=ray.tune.grid_search([1.]),
                    team_total_reward=True,
                    sector_config=dict(sqrt_min_area=250., sqrt_max_area=300.),
                    traffic_config=dict(buffer=nm2m(30.),
                                        min_distance=nm2m(80.),
                                        flight_config=dict(drift_penalty_coef=ray.tune.grid_search([.5]),
                                                            return_penalty=.0,
                                                            turn_rate_penalty_coef=ray.tune.grid_search([1.]),
                                                            ))))
        

        
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
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        landmark_pos = []
        for i, landmark in enumerate(world.landmarks):
            landmark_pos.append(landmark.state.p_pos.tolist())
        return (rew, collisions, min_dists, occupied_landmarks,landmark_pos)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward_ausiliary(self, agent, world):
        
        # reward defined as callback "flight_rew_callback"

        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0

        turn_rate_penalty_coef = self.config.get('turn_rate_penalty_coef', 1.0)
        drift_penalty_coef = self.config.get('drift_penalty_coef', 1.0)
        return_penalty = self.config.get('return_penalty', 0.0)

        efficiency_penalty = -np.abs(self.drift_obs / np.pi) * drift_penalty_coef
        self.efficiency_penalty += efficiency_penalty

        if self.action == 1:
            instruction_penalty = -return_penalty
        elif self.action > 1:
            instruction_penalty = -turn_rate_penalty_coef
        else:
            instruction_penalty = 0.

        self.instruction_penalty += instruction_penalty

        return instruction_penalty + efficiency_penalty
    
    # first call reward_ausiliary for each agent and put in the same data structure
    def realrewardforeverybody(reward_ausiliary()):
        rew_agents = set(rew.keys())

        safe_distance = self.config.get('safe_distance', nm2m(5.))
        conflict_penalty_coef = self.config.get('conflict_penalty_coef', 50.)
        conflict_penalty_decay = self.config.get('conflict_penalty_decay', 1.)
        alert_penalty_coef = self.config.get('alert_penalty_coef', 20.)
        alert_penalty_decay = self.config.get('alert_penalty_decay', 1.)
        alert_penalty_t_decay = self.config.get('alert_penalty_t_decay', 1.)
        team_reward = self.config.get('team_reward', True)

        result = {i: float(0.) for i in self.agent_ids}  # no penalty by default

        for i in rew_agents:
            conflict_penalty = 0.0
            alert_penalty = 0.0

            for j in rew_agents.difference(set([i])):
                dist = distance(self.agents[i].position, self.agents[j].position)
                conflict_penalty += logistic_penalty(conflict_penalty_coef, conflict_penalty_decay, m2nm(safe_distance),
                                                    m2nm(dist))

                cpa = self.agents[i].cpa(self.agents[j])
                d_cpa = cpa['d_pq']
                t_cpa = cpa['t']

                if t_cpa >= 0.:
                    t_decay = logistic_penalty(-1., alert_penalty_t_decay, 5., t_cpa / 60.)
                    alert_penalty += logistic_penalty(alert_penalty_coef, alert_penalty_decay, m2nm(safe_distance),
                                                    m2nm(d_cpa)) * t_decay

                    if t_cpa < 120. and d_cpa < safe_distance:
                        self.cum_alerts = self.cum_alerts.union(set([(i, j)]))

            result[i] = conflict_penalty + alert_penalty + rew[i]

            self.conflict_penalty += conflict_penalty
            self.alert_penalty += alert_penalty

        if team_reward:
            # TODO mean or sum!??
            total_reward = np.sum(list(result.values()))
            for i in self.agent_ids:
                result[i] = total_reward

        return dict(sorted(result.items()))



    def observation(self, agent, world):
    # flight_obs_callback(self):
    return np.array([self.length_obs / nm2m(100.),
                     np.cos(self.drift_obs),
                     np.sin(self.drift_obs),
                     self.v_ratio_obs])