import sys

from setuptools import setup, find_packages


if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running Python {}.'.format(sys.version_info.major))
    print('The installation will likely fail.')


setup(
    name='microtbs_rl',
    packages=[package for package in find_packages() if package.startswith('microtbs_rl')],
    install_requires=[
        'baselines>=0.1.4',
        'gym>=0.9.6',
        'tensorflow>=1.4.0',
        'pygame',
        'imageio',
        'matplotlib',
        'numpy',
    ],
    description='MicroTbs learning environment and the implementation of several RL algorithms',
    author='Alex Petrenko',
    url='https://github.com/alex-petrenko/microtbs-rl',
    author_email='apetrenko1991@gmail.com',
    version='1.0.0',
)
