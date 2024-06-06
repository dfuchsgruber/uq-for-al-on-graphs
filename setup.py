from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("graph_al/utils/_sbm.pyx"),
    include_dirs=[numpy.get_include()],
)