# Main Requirements
- Python 3.5
- Pytorch version < 0.4.0
- [multi-agent-particle-env](https://bitbucket.org/epesce/multi-agent-particle-env/src/master/)
 - TensorboardX
 - Numpy

# Usage

To launch experiments (training + evaluation) over a number of seeds just run this:
```
$ python main_seeds.py
```

Experiments settings can be changed editing the parameters file: ``params_seeds.py``


To use MADDPG set ``agent_alg = 'MADDPG`` in ``main_seeds.py``.
To use DDPG set ``agent_alg = 'DDPG`` in ``main_seeds.py``


## Contributors
Emanuele Pesce
