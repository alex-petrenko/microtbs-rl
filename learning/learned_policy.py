import argparse

import pygame

from agent_dqn import AgentDqn
from micro_tbs import Game, GameplayOptions
from utils import *

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
    pygame.init()

    args = parse_args()
    logger.info('Args: %r', args)
    train = args.train

    options = GameplayOptions(diagonal_moves=False)
    game = Game(options=options, windowless=train)
    state = game.reset()

    agent = AgentDqn(game.allowed_actions(), state)
    agent.initialize()

    num_episodes = 0
    total_env_steps = 0

    while not game.should_quit():
        state = game.reset()
        num_episodes += 1
        agent.on_new_episode()

        if num_episodes % 10 == 0:
            logger.info('Episode: %r, avg steps per episode %r', num_episodes, total_env_steps // num_episodes)

        while not game.is_over():
            if train:
                state = agent.explore(game, state)
            else:
                game.process_events()
                action = agent.act(state)
                state, _ = game.step(action)
                game.render()
                game.clock.tick(5)

        total_env_steps += game.num_steps

        if train:
            for i in range(15):
                agent.update()

    logger.info('Exiting...')


if __name__ == '__main__':
    sys.exit(main())
