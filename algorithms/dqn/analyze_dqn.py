import gym
import numpy as np
import matplotlib.pyplot as plt

import envs
from algorithms import dqn

from utils.dnn_utils import *
from utils.common_utils import *


logger = logging.getLogger(os.path.basename(__file__))


def deprocess_image(img):
    x = np.copy(img)
    mean, std = x.mean(), x.std()

    min_x = x.min()
    x -= min_x  # start at 0

    max_x = x.max()
    x /= (max_x - tf.keras.backend.epsilon())
    x *= 255  # max -> 255

    logger.info('mean: %r, std: %r, min: %r, max: %r', mean, std, min_x, max_x)

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (tf.sqrt(tf.reduce_mean(tf.square(x))) + tf.keras.backend.epsilon())


def test_target_loss(img):
    loss = tf.reduce_sum(img[:, :, 0] - img[:, :, 1] + img[:, :, 2])
    return loss


def image_loss(img):
    loss = tf.where(img > 1.0, tf.square(img), tf.zeros_like(img))
    loss += tf.where(img < 0.0, tf.square(img - 1), tf.zeros_like(img))
    loss = tf.reduce_sum(loss)
    return 0.001 * loss


def main():
    init_logger(os.path.basename(__file__))

    env = gym.make('MicroTbs-CollectSimple-v0')
    env.seed(0)
    observation = env.reset()
    env.render()

    experiment = 'dqn'
    params = dqn.AgentDqn.Params(experiment).load()
    agent = dqn.AgentDqn(env, params)
    agent.initialize()

    w, h, channels = env.observation_space.shape
    # input_img_data = np.random.random((w, h, channels))
    # input_img_data = (input_img_data - 0.5) * 20 + 128
    # input_img_data = np.full((w, h, channels), 0)
    # input_img_data[0, :, 0] = 255
    # input_img_data[0, :, 1] = 0
    # input_img_data[0, :, 2] = 0
    # input_img_data[:, 3, 0] = 255

    input_img_data = observation
    input_img_data = input_img_data.astype(np.float32)
    plt.imshow(deprocess_image(input_img_data))
    plt.show()

    # image = tf.placeholder(tf.float32, shape=[w, h, channels])
    image = agent.primary_dqn.observations
    target_loss = -agent.primary_dqn.Q_best

    image_penalty = image_loss(image)
    loss = target_loss + image_penalty

    gradient = tf.gradients(loss, image)
    gradient = normalize(gradient)

    session = agent.session
    step = 0.01
    best_loss = None
    for i in range(100000):
        target_loss_v, image_loss_v, grad = session.run(
            [target_loss, image_penalty, gradient], feed_dict={image: [input_img_data]},
        )

        best_loss = target_loss_v[0] if best_loss is None else min(best_loss, target_loss_v[0])

        if i % 10 == 0:
            logger.info('loss %r, image_loss %r, step %.2f', target_loss_v[0], image_loss_v, step)
            logger.info('best_loss %.3f', best_loss)
        input_img_data -= grad[0][0] * step
        step *= 0.9999

        if i % 10 == 0:
            plt.imshow(deprocess_image(input_img_data))
            plt.show()


if __name__ == '__main__':
    sys.exit(main())
