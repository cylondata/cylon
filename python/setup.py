# References
'''
https://github.com/FedericoStra/cython-package-example/blob/master/setup.py
https://github.com/thewtex/cython-cmake-example/blob/master/setup.py
'''

import os
import sysconfig
from distutils.sysconfig import get_python_lib
import pyarrow as pa
import numpy as np

from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

# os.environ["CXX"] = "mpic++"
pyarrow_location = os.path.dirname(pa.__file__)

ARROW_HOME = os.environ.get('ARROW_HOME')

try:
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0") or "0")
except Exception:
    nthreads = 0

compiler_directives = {"language_level": 3, "embedsignature": True}
cython_files = ["pytwisterx/*/*.pyx"]

if not ARROW_HOME:
    raise ValueError("ARROW_HOME not set")

# For now, assume that we build against bundled pyarrow releases.

std_version = '-std=c++14'
pyarrow_include_dir = os.path.join(pyarrow_location, 'include')
extra_compile_args = os.popen("mpic++ --showme:compile").read().strip().split(' ')
extra_link_args = os.popen("mpic++ --showme:link").read().strip().split(' ')
additional_compile_args = [std_version,
                           '-DARROW_METADATA_V4 -DGOOGLE_GLOG_DLL_DECL="" -DNEED_EXCLUSIVE_SCAN']
extra_compile_args = extra_link_args + additional_compile_args
extra_link_args.append("-Wl,-rpath,$ORIGIN/pyarrow")

arrow_library_directory = os.path.join(ARROW_HOME, "arrow", "install", "lib")
arrow_lib_include_dir = os.path.join(ARROW_HOME, "arrow", "install", "include")
twisterx_library_directory = os.path.join(ARROW_HOME, "lib")

library_directories = [twisterx_library_directory,
                       arrow_library_directory,
                       get_python_lib(),
                       os.path.join(os.sys.prefix, "lib")]

libraries = ["arrow", "twisterx", "twisterx_python", "glog"]

_include_dirs = ["../cpp/src/twisterx/python",
                 "../cpp/src/twisterx/lib",
                 "../cpp/src/twisterx/",
                 "../cpp/src/twisterx/net",
                 "../cpp/src/twisterx/data",
                 "../cpp/src/twisterx/io",
                 "../cpp/src/twisterx/join",
                 "../cpp/src/twisterx/util",
                 arrow_library_directory,
                 arrow_lib_include_dir,
                 pyarrow_include_dir,
                 np.get_include(),
                 ]

# Adopted the Cudf Python Build format
# https://github.com/rapidsai/cudf

extensions = [
    Extension(
        "*",
        sources=cython_files,
        include_dirs=_include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        library_dirs=library_directories,
    )
]

compiler_directives = {"language_level": 3, "embedsignature": True}
packages = find_packages(include=["pytwisterx", "pytwisterx.*"])

setup(
    name="pytwisterx",
    packages=packages,
    version='0.0.1',
    setup_requires=["cython",
                    "setuptools",
                    "numpy",
                    ],
    ext_modules=cythonize(
        extensions,
        nthreads=nthreads,
        compiler_directives=dict(
            profile=False, language_level=3, embedsignature=True
        ),
    ),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'cython',
        'pyarrow==0.16.0'
    ],
    zip_safe=False,
)
