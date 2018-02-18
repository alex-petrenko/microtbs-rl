# Simple Reinforcement Learning

This repository contains:
* Implementation of the [gym](https://github.com/openai/gym)-compatible learning environment called MicroTbs (for Micro Turn-Based Strategy)
* Reinforcement learning algorithms aimed to solve this game
  * DQN algorithm, inspired by the original DeepMind's
    [work](https://www.nature.com/articles/nature14236 "Deep Mind's Nature Paper (similar work can be found on arxiv)")
  * Synchronous Advantage Actor-Critic (A2C),
    based on the original [A3C](https://arxiv.org/pdf/1602.01783.pdf) algorithm,
    and [OpenAI baselines](https://github.com/openai/baselines) implementation.

The implementation of RL algorithms is pretty straightforward to keep
the code relatively simple and easy to understand. Yet no assumptions are made
regarding the environment, therefore the code can be used to solve other tasks, not just this gridworld.

**Here's how it looks**

1. A2C solves simple version of the environment _MicroTbs-CollectWithTerrain-v2_:

![a2c_terrain](https://github.com/alex-petrenko/simple-reinforcement-learning/blob/master/misc/a2c_terrain_visualization.gif?raw=true)


_To be continued..._
