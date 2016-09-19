from setuptools import setup, find_packages

setup(
    name='voice2vec',
    version='0.0.1',
    description='Voice to vector. Voice similarity',
    license='MIT',
    url='https://github.com/xenx/speech',

    install_requires=['flask', 'numpy', 'librosa', 'theano', 'lasagne', 'dill'],

    package=find_packages(exclude=['web']),
)
