import sys
import os
import importlib
import distutils.core
import requests

install_requires = [
    'ultralytics',
    'argparse',
]

distutils.core.setup(install_requires=install_requires)
