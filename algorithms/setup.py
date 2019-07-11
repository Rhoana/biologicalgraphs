from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

graph_software_dir = '{}/software/'.format(os.path.expanduser('~'))

extensions = [
    Extension(
        name='multicut',
        include_dirs=[np.get_include(), '{}/graph/include'.format(graph_software_dir)],
        sources=['multicut.pyx', 'cpp-multicut.cpp'],
        extra_compile_args=['-O4', '-std=c++11'],
        language='c++'
    ),
    Extension(
        name='lifted_multicut',
        include_dirs=[np.get_include(), '{}/graph/include'.format(graph_software_dir)],
        sources=['lifted_multicut.pyx', 'cpp-lifted-multicut.cpp'],
        extra_compile_args=['-O4', '-std=c++11'],
        language='c++'
    )
]

setup(
    name='algorithms',
    ext_modules = cythonize(extensions)
)
