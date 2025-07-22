
import platform
import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

if platform.system() == 'Darwin':
  include_dirs = ['/opt/local/libexec/boost/1.81/include', np.get_include()]
elif platform.system() == 'Linux':
  include_dirs = ['../boost_1_88_0', np.get_include()]

setup(
    ext_modules = cythonize([
    Extension('pymmr.legendre',
              sources=["pymmr/legendre.pyx"],                 # our Cython source
              include_dirs=include_dirs,
              language="c++",             # generate C++ code
             ),
    ], language_level=3)
)
