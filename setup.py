#!/usr/bin/env python

from distutils.core import setup

setup(name='georgefit',
      version='0.1',
      description='Fit data with GPs',
      author='João Faria',
      author_email='joao.faria@astro.up.pt',
    #   url='https',
      packages=['georgefit', 'georgefit.datasets'],
      package_dir={'georgefit': 'georgefit'},
      package_data={'georgefit': ['datasets/*.dat']},
     )