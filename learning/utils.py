import os
import sys
import time
import logging

from os.path import join


EPS = 1e-5

LOGGING_FOLDER = 'C:/temp/py_logging'


def bp():
    import ipdb; ipdb.set_trace()

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
