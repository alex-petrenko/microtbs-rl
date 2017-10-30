import os
import sys
import math
import time
import logging

from os.path import join


EPS = 1e-5
LOGGING_FOLDER = 'C:/temp/py_logging'  # use environment variable instead?


class Vec:
    def __init__(self, i, j):
        self.i = i
        self.j = j

    @property
    def x(self):
        return self.j

    @x.setter
    def x(self, x):
        self.j = x

    @property
    def y(self):
        return self.i

    @y.setter
    def y(self, y):
        self.i = y

    @property
    def ij(self):
        return self.i, self.j

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.i, self.j))

    def __neg__(self):
        return Vec(-self.i, -self.j)

    def __add__(self, other):
        return Vec(self.i + other.i, self.j + other.j)

    def __sub__(self, other):
        return self + (-other)

    def dist_sq(self, other):
        return (self.i - other.i) ** 2 + (self.j - other.j) ** 2

    def dist(self, other):
        return math.sqrt(self.dist_sq(other))


# Helper functions

def bp():
    import ipdb
    ipdb.set_trace()


def init_logger(script_name):
    """Initialize logging facilities for particular script."""
    log_folder = join(LOGGING_FOLDER, script_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_filename = '{time}_{name}.log'.format(
        time=time.strftime("%Y_%m_%d_%H_%M_%S"), name=script_name,
    )
    log_path = join(log_folder, log_filename)

    fmt_str = '%(asctime)s.%(msecs)03d %(name)s:%(lineno)d %(levelname)s %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt_str,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_path,
        filemode='w',
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=fmt_str, datefmt='%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info('Logging initialized!')
