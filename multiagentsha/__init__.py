from gym.envs.registration import register
import multiagentsha.scenarios as scenarios
# Multiagent envs
# ----------------------------------------


_particles = {
    "navigation" : "Navigation-v0",
    "navigation_red_rew": "Navigation-rew-v0",
    "navigation_red_rew2": "Navigation-rew0.1-v0",
    "navigation_red_rew3": "Navigation-rew0.25-v0",
    "navigation_red_rew_coll": "Navigation-rew-col-v0"
#    "multi_speaker_listener": "MultiSpeakerListener-v0",
#    "simple_adversary": "SimpleAdversary-v0",
#    "simple_crypto": "SimpleCrypto-v0",
#    "simple_push": "SimplePush-v0",
#    "simple_reference": "SimpleReference-v0",
#    "simple_speaker_listener": "SimpleSpeakerListener-v0",
#    "simple_spread": "SimpleSpread-v0",
#    "simple_tag": "SimpleTag-v0",
#    "simple_world_comm": "SimpleWorldComm-v0",
#    "climbing_spread": "ClimbingSpread-v0",
}

for scenario_name, gymkey in _particles.items():
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()

    # Registers multi-agent particle environments:
    register(
        gymkey,
        entry_point="multiagentsha.environment:MultiAgentOrigEnv",
        kwargs={
            "discrete_action": True,
            "world": world,
            "reset_callback": scenario.reset_world,
            "info_callback": scenario.info_callback,
            "reward_callback": scenario.reward,
            "observation_callback": scenario.observation,
        },
    )


