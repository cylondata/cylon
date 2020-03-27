# References
'''
https://github.com/FedericoStra/cython-package-example/blob/master/setup.py
https://github.com/thewtex/cython-cmake-example/blob/master/setup.py
'''

import os
from setuptools import setup, find_packages, find_namespace_packages, Extension
from Cython.Build import cythonize

import numpy
import pyarrow

mpi_compile_args = os.popen("mpic++ --showme:compile").read().strip().split(' ')
mpi_link_args = os.popen("mpic++ --showme:link").read().strip().split(' ')

ext_modules = [
    Extension("pytwisterx.geometry",
              sources=["../cpp/src/lib/Circle.cpp", "twisterx/geometry/circle.pyx"],
              include_dirs=[numpy.get_include(), "../cpp/src/lib", pyarrow.get_include()],
              language='c++',
              extra_compile_args=mpi_compile_args,
              extra_link_args=mpi_link_args,
              )
]

compiler_directives = {"language_level": 3, "embedsignature": True}
ext_modules = cythonize(ext_modules, compiler_directives=compiler_directives)

setup(
    name="pytwisterx",
    packages=['twisterx', 'twisterx.geometry'],
    version='0.0.1',
    ext_modules=ext_modules,
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'cython',
        'pyarrow'
    ],
)

print(find_namespace_packages())
