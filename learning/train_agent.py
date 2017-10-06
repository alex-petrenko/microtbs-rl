import os
import sys
import logging

from utils import *

from agent import AgentRandom
from turn_based_strategy import TurnBasedStrategy


logger = logging.getLogger(os.path.basename(__file__))


def main():
    """Script entry point."""
    init_logger(os.path.basename(__file__))

    game = TurnBasedStrategy()
    agent = AgentRandom(game.allowed_actions())

    while not game.should_quit():
        state = game.reset()

        while not game.is_over():
            game.process_events()
            action = agent.act(state)
            state = game.step(action)
            game.render()
            game.clock.tick(10)

    logger.info('Exiting...')


if __name__ == '__main__':
    sys.exit(main())
