import os
import sys
import time
import logging

from os.path import join


EPS = 1e-5

if sys.platform == 'win32':
    LOGGING_FOLDER = 'C:/temp/py_logging'  # use environment variable instead?
else:
    LOGGING_FOLDER = '/tmp/py_logging'

LOGGING_INITIALIZED = False


# Filesystem helpers

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def project_root():
    return os.path.dirname(os.path.dirname(__file__))


def experiments_dir():
    return ensure_dir_exists(join(project_root(), '.experiments'))


def experiment_dir(experiment):
    return ensure_dir_exists(join(experiments_dir(), experiment))


def model_dir(experiment):
    return ensure_dir_exists(join(experiment_dir(experiment), '.model'))


def stats_dir(experiment):
    return ensure_dir_exists(join(experiment_dir(experiment), '.stats'))


def summaries_dir():
    return ensure_dir_exists(join(project_root(), '.summary'))


# Keeping track of experiments

def get_experiment_name(env_id, name):
    return '{}-{}'.format(env_id, name)


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


def get_test_logger():
    global LOGGING_INITIALIZED
    if LOGGING_INITIALIZED is False:
        init_logger('test')
        LOGGING_INITIALIZED = True

    return logging.getLogger('test')
