import csv
import numpy as np
import matplotlib.pyplot as plt

from utils.common_utils import *
from utils.monitor import Monitor


COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue', 'yellow']

logger = logging.getLogger(os.path.basename(__file__))


def main():
    init_logger(os.path.basename(__file__))

    experiments_v0 = [
        get_experiment_name('MicroTbs-CollectWithTerrain-v0', 'dqn_v0'),
        get_experiment_name('MicroTbs-CollectWithTerrain-v0', 'dqn_v3_inception'),

        get_experiment_name('MicroTbs-CollectWithTerrain-v0', 'a2c_v0'),
        get_experiment_name('MicroTbs-CollectWithTerrain-v0', 'a2c_v1'),
        get_experiment_name('MicroTbs-CollectWithTerrain-v0', 'a2c_v2_inception'),
        get_experiment_name('MicroTbs-CollectWithTerrain-v0', 'a2c_v4_10steps_097'),

        get_experiment_name('MicroTbs-CollectWithTerrain-v0', 'openai_dqn'),
    ]

    experiments_terrain_a2c = [
        get_experiment_name('MicroTbs-CollectWithTerrain-v1', 'a2c_v0'),
        get_experiment_name('MicroTbs-CollectWithTerrain-v1', 'a2c_v1'),
        get_experiment_name('MicroTbs-CollectWithTerrain-v2', 'a2c_v5'),
    ]

    experiments_partially_observable = [
        get_experiment_name('MicroTbs-CollectPartiallyObservable-v1', 'a2c_v2'),
        get_experiment_name('MicroTbs-CollectPartiallyObservable-v1', 'a2c_v3'),
        get_experiment_name('MicroTbs-CollectPartiallyObservable-v3', 'a2c_v5'),
    ]

    experiments = experiments_terrain_a2c

    for i in range(len(experiments)):
        experiment = experiments[i]
        x, y = [], []
        stats_filename = Monitor.stats_filename(experiment)
        with open(stats_filename) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                x.append(int(row[0]))
                y.append(float(row[1]))

        skip_coeff = max(1, len(x) // 200)

        x_filtered, y_filtered = [], []
        for j in range(len(x)):
            if j % skip_coeff == 0:
                x_filtered.append(x[j])
                y_filtered.append(y[j])

        x = x_filtered
        y = y_filtered

        logger.info('Plotting %s...', experiment)
        plt.plot(x, y, color=COLORS[i], label=experiment)

    plt.title('Reward over time')
    plt.xlabel('Training step (batch #)')
    plt.ylabel('Mean reward')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sys.exit(main())
