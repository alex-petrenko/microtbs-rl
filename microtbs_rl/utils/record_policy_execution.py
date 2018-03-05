"""
Render the policy execution to gif or video.
Gifs are generated with pure python code.
To render a video, it just saves a bunch of images to a directory. Then it's very easy to render a video with ffmpeg.
Like: ffmpeg -framerate 8 -pattern_type glob -i '*.png' out.mp4

"""

import gym
import imageio
import matplotlib.pyplot as plt

from microtbs_rl.envs import micro_tbs

from microtbs_rl.utils.common_utils import *

from microtbs_rl.algorithms import a2c


logger = logging.getLogger(os.path.basename(__file__))


def record(experiment, env_id, save_as_gif=False, num_episodes=100, fps=6):
    env = gym.make(env_id)
    env.render_resolution = 1080
    env.seed(3)

    params = a2c.AgentA2C.Params(experiment).load()
    agent = a2c.AgentA2C(env, params)
    agent.initialize()
    footage_dir = join(experiment_dir(experiment), '.footage')
    ensure_dir_exists(footage_dir)

    # if we render a shorter gif, let's just keep all the frames in memory
    game_screens = [] if save_as_gif else None

    def render_and_save_game_screen(_episode, _step):
        game_screen = env.render(mode='rgb_array')
        img_filename = join(footage_dir, '{episode:05d}_{step:05d}.png'.format(episode=_episode, step=_step))
        plt.imsave(img_filename, game_screen)
        if game_screens is not None:
            game_screens.append(game_screen)

    for episode_idx in range(num_episodes):
        logger.info('Episode #%d', episode_idx)

        # Make sure the generated environment is challenging enough for our agent (to make it interesting to watch).
        # Re-generate new worlds until the right conditions are met.
        while True:
            obs = env.reset()
            border_num_obstacles = env.world_size ** 2 - env.mode.play_area_size ** 2
            num_obstacles = sum(isinstance(t, micro_tbs.Obstacle) for t in env.terrain.flatten())
            min_obstacles_in_play_area = 7
            min_gold_piles = 4

            enough_obstacles = num_obstacles >= border_num_obstacles + min_obstacles_in_play_area
            enough_gold = env.num_gold_piles >= min_gold_piles
            env_is_interesting = enough_gold and enough_obstacles
            if env_is_interesting:
                break

        step = 0
        done = False
        while not done:
            render_and_save_game_screen(episode_idx, step)
            action = agent.best_action(obs)
            obs, _, done, _ = env.step(action)
            step += 1

        render_and_save_game_screen(episode_idx, step)

    agent.finalize()
    env.close()

    if save_as_gif:
        logger.info('Rendering gif...')
        gif_name = join(footage_dir, '{}.gif'.format(experiment))
        kwargs = {'duration': 1.0 / fps}
        imageio.mimsave(gif_name, game_screens, 'GIF', **kwargs)

    return 0


def main():
    init_logger(os.path.basename(__file__))
    env_id = a2c.a2c_utils.CURRENT_ENV
    experiment = get_experiment_name(env_id, a2c.a2c_utils.CURRENT_EXPERIMENT)
    return record(experiment, env_id)


if __name__ == '__main__':
    sys.exit(main())
