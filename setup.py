# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import diversityrgc

setup(name='diversityrgc',
      version=diversityrgc.__version__,
      description='Functional caracterization of retinal ganglion cell diversity tool',
      author='Rubén Crespo-Cano',
      author_email='rcrespocano@gmail.com',
      url='https://github.com/rcrespocano/functional-diversity-rgc.git',
      long_description='Functional caracterization of retinal ganglion cell diversity tool',
      packages=find_packages(),
)
