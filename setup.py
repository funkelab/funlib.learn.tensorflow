from Cython.Build import cythonize
from distutils.sysconfig import get_python_inc, get_config_var
from setuptools import setup
from setuptools.extension import Extension
import numpy as np
import os

print(np.get_include())

losses_impl_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'funlib',
    'learn',
    'tensorflow',
    'losses',
    'impl')

include_dirs = [
    losses_impl_dir,
    os.path.dirname(get_python_inc()),
    get_python_inc(),
    np.get_include()
]

library_dirs = [
    losses_impl_dir,
    get_config_var("LIBDIR")
]

setup(
        name='funlib.learn.tensorflow',
        version='0.1',
        url='https://github.com/funkelab/funlib.learn.tensorflow',
        author='Jan Funke',
        author_email='funkej@janelia.hhmi.org',
        license='MIT',
        packages=[
            'funlib.learn.tensorflow',
            'funlib.learn.tensorflow.models',
            'funlib.learn.tensorflow.losses',
            'funlib.learn.tensorflow.losses.impl',
        ],
        ext_modules=cythonize([
            Extension(
                'funlib.learn.tensorflow.losses.impl.wrappers',
                sources=[
                    'funlib/learn/tensorflow/losses/impl/wrappers.pyx',
                    'funlib/learn/tensorflow/losses/impl/emst.cpp',
                    'funlib/learn/tensorflow/losses/impl/um_loss.cpp',
                ],
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                extra_link_args=['-std=c++11', "-lmlpack"],
                extra_compile_args=['-O3', '-std=c++11'],
                language='c++')
        ])
)
