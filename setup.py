from distutils.core import setup
from Cython.Build import cythonize
import numpy 


# Compilar o arquivo para C
setup(ext_modules = cythonize('Models.pyx'),
    include_dirs = [numpy.get_include()] )