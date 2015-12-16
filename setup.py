'''A setuptools based installer for 163223dpose.

Based on https://github.com/pypa/sampleproject/blob/master/setup.py

Matt Vernacchia
2015 Dec 16
'''

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='163223dpose',
    
    version='0.0.0a0',

    description='16.322 final project on 3D pose estimation.',

    author='Matt Vernacchia',
    author_email='mvernacc@mit.edu.',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
    ],

     # What does your project relate to?
    keywords='kalman UKF quaternion estimation',


    install_requires=['numpy', 'scipy', 'matplotlib', 'transforms3d'],

    packages=find_packages(),

    scripts=[],

)
