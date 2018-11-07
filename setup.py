from os import path
from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setup(
    name='autoclust',
    version='0.0.9000',
    description='Automated clustering with Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Christopher Baker',
    author_email='chriscrewbaker@gmail.com',
    license='LICENSE.txt',
    packages=['autoclust'],
    install_requires=['scikit-learn'],
    setup_requires=["pytest-runner"]
)
