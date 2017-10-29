import pygame

from micro_tbs import Game, GameplayOptions
from utils import *

logger = logging.getLogger(os.path.basename(__file__))


def main():
    """Script entry point."""
    init_logger(os.path.basename(__file__))
    pygame.init()

    gameplay_options = GameplayOptions.collect_gold_simple()
    game = Game(gameplay_options)

    while not game.should_quit():
        game.reset()

        while not game.is_over():
            action = game.process_events()
            game.step(action)
            game.render()

            game.clock.tick(30)

    logger.info('Exiting...')


if __name__ == '__main__':
    sys.exit(main())
