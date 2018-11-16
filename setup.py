from os import path
from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setup(
    name='validclust',
    version='0.0.9000',
    description='Validate clustering results',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Christopher Baker',
    author_email='chriscrewbaker@gmail.com',
    license='LICENSE.txt',
    packages=['validclust'],
    install_requires=['scikit-learn', 'pandas', 'numpy'],
    tests_require=['pytest'],
    setup_requires=["pytest-runner"]
)
