import os
import sys
import logging

from utils import *

from turn_based_strategy import Game


logger = logging.getLogger(os.path.basename(__file__))


def main():
    """Script entry point."""
    init_logger(os.path.basename(__file__))

    game = Game()

    while not game.should_quit():
        game.reset()

        while not game.is_over():
            action = game.process_events()
            logger.info('Action %d', action)
            game.step(action)
            game.render()

            game.clock.tick(30)

    logger.info('Exiting...')


if __name__ == '__main__':
    sys.exit(main())
