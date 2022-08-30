from pathlib import Path

from setuptools import find_packages, setup

NAME = 'spaceship_titanic_classification_model'
DESCRIPTION = 'This is an toy project to learn the basics of MLOps and software design in the machine learning context'
URL = 'https://github.com/kojr1234/mlops_exercise.git'
EMAIL = 'fujii.kimio.k@gmail.com'
AUTHOR= 'FabioFujii'
REQUIRES_PYTHON = '>=3.9.0'

long_description = DESCRIPTION

# load the package's version fila as a dictionary
about = {}
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / 'requirements'
PACKAGE_DIR = ROOT_DIR / 'classifier_model'
with open(PACKAGE_DIR / 'VERSION', 'r') as f:
    about['__version__'] = f.read().strip()

