"""
Helper context manager, saves the training progress to .csv file.
Very useful for comparing different algorithms.
Plot the .cvs file with plotter.py

"""


import datetime

from microtbs_rl.utils.common_utils import *


class Monitor:
    def __init__(self, experiment):
        self.experiment = experiment
        self.progress_file = None

    def __enter__(self):
        stats_filename = self.stats_filename(self.experiment)
        if os.path.isfile(stats_filename):
            tstamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            os.rename(stats_filename, '{}_old_{}'.format(stats_filename, tstamp))

        self.progress_file = open(self.stats_filename(self.experiment), 'w')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.progress_file.close()

    @staticmethod
    def stats_filename(experiment):
        return join(stats_dir(experiment), 'progress.csv')

    def callback(self, local_vars, _):
        """Override in the derived class."""
        pass
