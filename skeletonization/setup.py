from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name='generate_skeletons',
        include_dirs=[np.get_include()],
        sources=['generate_skeletons.pyx', 'cpp-thinning.cpp', 'cpp-upsample.cpp'],
        extra_compile_args=['-O4', '-std=c++0x'],
        language='c++'
    )
]

setup(
    name='skeletonization',
    ext_modules=cythonize(extensions)
)
