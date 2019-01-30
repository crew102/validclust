import sys
from setuptools import setup
from os import path

install_requires = ['scikit-learn', 'pandas', 'numpy', 'seaborn', 'matplotlib']

is_v2 = sys.version_info[0] == 2
is_low_v3 = sys.version_info[0] == 3 and sys.version_info[1] <= 4

if is_v2 or is_low_v3:
    install_requires.remove('matplotlib')
    install_requires.append('matplotlib<3')

if is_v2:
    install_requires.append('scipy<1.2')

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setup(
    name='validclust',
    version='0.0.0.9000',
    description='Validate clustering results',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Christopher Baker',
    author_email='chriscrewbaker@gmail.com',
    license='LICENSE.txt',
    packages=['validclust'],
    install_requires=install_requires,
    tests_require='pytest',
    setup_requires='pytest-runner'
)
