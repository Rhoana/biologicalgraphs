from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name='node_generation',
        include_dirs=[np.get_include()],
        sources=['node_generation.pyx', 'cpp-node-generation.cpp'],
        extra_compile_args=['-O4', '-std=c++0x'],
        language='c++'
    )
]

setup(
    name='node_generation',
    ext_modules=cythonize(extensions)
)