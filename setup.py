from setuptools import setup

with open('requirements.txt', 'r') as file:
    requirements = [line.strip() for line in file if line.strip()]

setup(
    name='NLP assignment 2',
    description="Implementation of an emotion classifier.",
    install_requires=requirements,
)
