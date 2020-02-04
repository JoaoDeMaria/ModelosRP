from distutils.core import setup
from Cython.Build import cythonize
import numpy 


# Compilar o arquivo para C
setup(ext_modules = cythonize('utils.pyx'),
    include_dirs = [numpy.get_include()] )