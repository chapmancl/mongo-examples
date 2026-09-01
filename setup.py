from setuptools import find_packages, setup

setup(
    name='gem-mongodb-ai',
    version='1.0',
    packages=find_packages(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'coverage']
)
