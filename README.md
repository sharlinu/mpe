# Summary
This is an alternative maddpg implementation in pytorch that also takes in episodic tasks and does not rely on old package versions anymore

# Requirements used 
- Python 3.9
- Pytorch 1.10.2
- [multi-agent-particle-env](https://bitbucket.org/epesce/multi-agent-particle-env/src/master/)
- TensorboardX
- Numpy 1.24.0
- Baselines

# Usage
To launch experiments (training + evaluation) over a number of seeds just run this:
```
$ python main_seeds.py
```

Experiments settings can be changed editing the parameters file: ``params_seeds.py``

To use MADDPG set ``agent_alg = 'MADDPG`` in ``main_seeds.py``.
To use DDPG set ``agent_alg = 'DDPG`` in ``main_seeds.py``
