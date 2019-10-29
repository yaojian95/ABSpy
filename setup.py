# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

exec(open('abs/version.py').read())

setup(name="abs",
      version=__version__,
      description="Analytical method of Blind Separation",
      license="GPLv3",
      url="https://github.com/gioacchinowang/ABS",
      packages=find_packages(),
      dependency_links=[],
      python_requires='>=3.5',
      zip_safe=False,
      classifiers=["Development Status :: 4 - Beta",
                   "Topic :: Utilities",
                   "License :: OSI Approved :: GNU General Public License v3 "
                   "or later (GPLv3+)"],)
