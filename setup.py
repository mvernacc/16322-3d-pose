try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': '16.322 final project on 3D pose estimation.',
    'author': 'Matt Vernacchia and Mike Klinker',
    'url': 'URL to get it at.',
    'download_url': 'https://github.mit.edu/mvernacc/16322-3d-pose',
    'author_email': 'mvernaccd@mit.edu.',
    'version': '0.1',
    'install_requires': ['numpy', 'scipy', 'transforms3d'],
    'packages': ['sensor_drivers', 'estimators'],
    'scripts': [],
    'name': '163223dpose'
}

setup(**config)