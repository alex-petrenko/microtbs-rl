import os
import sys
import logging
import argparse

from utils import *

from turn_based_strategy import Game
from agent_dqn import AgentDqn, preprocess_state


logger = logging.getLogger(os.path.basename(__file__))


def parse_args():
    """Parse command line args using argparse."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train policy (rather than playback)',
    )
    return parser.parse_args()


def main():
    """Script entry point."""
    init_logger(os.path.basename(__file__))

    args = parse_args()
    logger.info('Args: %r', args)
    train = args.train

    game = Game()
    state = game.reset()

    agent = AgentDqn(game.allowed_actions(), preprocess_state(state))
    agent.initialize()

    num_episodes = 0
    while not game.should_quit():
        state = game.reset()
        num_episodes += 1
        logger.info('Episode: %r', num_episodes)
        while not game.is_over():
            game.process_events()

            if train:
                state = agent.explore(game, state)
                if num_episodes % 5 == 0:
                    agent.update()
            else:
                action = agent.act(state)
                state, _ = game.step(action)
                game.render()
                game.clock.tick(5)

    logger.info('Exiting...')


if __name__ == '__main__':
    sys.exit(main())
