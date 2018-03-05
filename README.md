## RL algorithms for the MicroTbs learning environment

This repository contains:
* Implementation of the [gym](https://github.com/openai/gym)-compatible learning environment called MicroTbs
(short for Micro Turn-Based Strategy)
* Reinforcement learning algorithms aimed to solve this game:
  * DQN algorithm, inspired by the original DeepMind's
    [work](https://www.nature.com/articles/nature14236 "Deep Mind's Nature Paper (similar work can be found on arxiv)")
  * Synchronous Advantage Actor-Critic (A2C),
    based on the original [A3C](https://arxiv.org/pdf/1602.01783.pdf) algorithm,
    and [OpenAI baselines](https://github.com/openai/baselines) implementation.

This repository was created for educational purposes,
to try and learn different RL techniques.
It may be useful for people who also want to experiment with deep RL
or just seek simple and easy-to-understand implementations of existing RL algorithms

**Here's how it looks**

1. A2C solves simple version of the environment _MicroTbs-CollectWithTerrain-v2_:

    ![a2c_terrain](https://github.com/alex-petrenko/rl-experiments/blob/master/misc/a2c_terrain_visualization.gif?raw=true)

    _a longer animation: <https://youtu.be/JykBihC0TvM>_

2. Feed-forward A2C in a more challenging version of the environment _MicroTbs-CollectPartiallyObservable-v3_ (clickable):

    [![Feed-forward A2C](https://img.youtube.com/vi/CP94lSM0zGM/0.jpg)](https://youtu.be/CP94lSM0zGM)

As seen in this example, a simple feed-forward method underperforms on this task, mostly due to following reasons:

* Feed-forward architecture does not "remember" where the agent has been on previous steps,
thus it regularly gets stuck and fails to explore unseen parts of the environment.
This can be solved by adding some memory to the policy network (like LSTM cells) and training it as a recurrent net.

* Agent has only visual input, the numeric information like the number of remaining movepoints isn't passed to the agent.
Therefore, it is not able to plan it's actions optimally in many cases.

* Quite curiously, the agent avoids "Stables" (brown object that gives additional movepoints).
This is caused by the tiny negative reward that agent receives for making each step that does not collect gold.
Early in the training, when the agent moves mostly randomly, taking stables means significantly increasing the
penalty received by the end of the episode.

* The agent's policy is stochastic, and during training the policy net is penalized for
having too low entropy of the probability distribution of actions. This encourages exploration
and prevents early convergence to sub-optimal policies, but sometimes can lead to seemingly stupid
behavior during the policy execution. Maybe it's a good idea to decay the entropy penalty during training.

* Due to the nature of the A2C algorithm, the agent fails to plan sufficiently far ahead,
and thus often gets stuck in obstacles. This can be solved by something
like a Monte-Carlo playout into the (predicted)
future or an [imagination module](https://arxiv.org/abs/1707.06203).


### About the environment

This environment was created as a playground, to experiment with various RL algorithms.
It resembles some of the traits of certain turn-based strategies like HOMM3, hence the name.
The task is to collect as much resources (gold) as possible with a limited number of movepoints.

Cell types in the environment:

* Red - a hero
* Yellow - gold piles
* Grey - walls, obstacles
* Green - swamp, increases movepoint penalty per move
* Brown - stables, increase hero's movepoints
* Light blue - lookout tower, opens the map in a certain radius
* Black - fog-of-war, unknown territory

Versions of the environment:

* CollectSimple - plain gold collection, no terrain or obstacles
* CollectWithTerrain - same, but with walls and obstacles in play area
* CollectPartiallyObservable - with all types of cells, map is bigger than view size
and must be explored

There's also a PvP version of the environment, that allows experiments with self-play, but it is unfinished.

### How-to

Play the environment by yourself, with human controls:

```shell
python -m microtbs_rl.envs.gameplay
```

Train a DQN agent with default parameters and see how it works:

```shell
python -m microtbs_rl.algorithms.dqn.train_dqn
python -m microtbs_rl.algorithms.dqn.enjoy_dqn
```

Train an A2C agent with default parameters and see how it works:

```shell
python -m microtbs_rl.algorithms.a2c.train_a2c
python -m microtbs_rl.algorithms.a2c.enjoy_a2c
```

Train a baseline OpenAI DQN implementation and see how it works:

```shell
python -m microtbs_rl.algorithms.baselines.openai_baselines.train_baseline_dqn
python -m microtbs_rl.algorithms.baselines.openai_baselines.enjoy_baseline_dqn
```

To compare performance and learning curves of different algorithms you can
modify the file plotter.py to add the experiments you're interested
in. Then just run:

```shell
python -m microtbs_rl.utils.plotter
```

Run unit tests:

```shell
python -m unittest
```

You can install this package into your python env and use it as a dependency:

```shell
pip install -e .
```


If you have any questions or problems please feel free to reach me: apetrenko1991@gmail.com
Or just go ahead and open an issue here on Github.
