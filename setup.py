import glob
import os

import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

_BOOST_HEADER = os.path.join('boost', 'math', 'special_functions', 'legendre.hpp')


def boost_include():
    """Directory holding the Boost headers: $BOOST_INCLUDE, or the first standard location that has them."""
    if 'BOOST_INCLUDE' in os.environ:
        return os.environ['BOOST_INCLUDE']

    def version(path):
        # /opt/local/libexec/boost/1.88/include -> (1, 88)
        return tuple(int(v) for v in path.split(os.sep)[-2].split('.') if v.isdigit())

    candidates = sorted(glob.glob('/opt/local/libexec/boost/*/include'), key=version, reverse=True)   # MacPorts
    candidates += ['/opt/local/include', '/opt/homebrew/include', '/usr/local/include', '/usr/include']
    candidates += sorted(glob.glob('../boost_1_*'), reverse=True)    # source tree next to the repository
    for d in candidates:
        if os.path.isfile(os.path.join(d, _BOOST_HEADER)):
            return d
    raise RuntimeError('Boost headers not found: set the environment variable BOOST_INCLUDE to the directory '
                       'that contains boost/math/special_functions/legendre.hpp')


setup(
    ext_modules=cythonize([
        Extension('pymmr.legendre',
                  sources=["pymmr/legendre.pyx"],                 # our Cython source
                  include_dirs=[boost_include(), np.get_include()],
                  language="c++",             # generate C++ code
                  ),
    ], language_level=3)
)
