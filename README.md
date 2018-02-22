# Reinforcement Learning Experiments

This repository contains:
* Implementation of the [gym](https://github.com/openai/gym)-compatible learning environment called MicroTbs (for Micro Turn-Based Strategy)
* Reinforcement learning algorithms aimed to solve this game
  * DQN algorithm, inspired by the original DeepMind's
    [work](https://www.nature.com/articles/nature14236 "Deep Mind's Nature Paper (similar work can be found on arxiv)")
  * Synchronous Advantage Actor-Critic (A2C),
    based on the original [A3C](https://arxiv.org/pdf/1602.01783.pdf) algorithm,
    and [OpenAI baselines](https://github.com/openai/baselines) implementation.

The implementation of RL algorithms is rather straightforward to keep
the code relatively simple and easy to understand. Yet no assumptions are made
about the environment, therefore the code can be used to solve arbitrary tasks, not just this gridworld.

**Here's how it looks**

1. A2C solves simple version of the environment _MicroTbs-CollectWithTerrain-v2_:

![a2c_terrain](https://github.com/alex-petrenko/rl-experiments/blob/master/misc/a2c_terrain_visualization.gif?raw=true)

2. Feed-forward A2C in more challenging version of the environment 

_To be continued..._
