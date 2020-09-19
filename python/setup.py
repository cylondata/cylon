##
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 # http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 ##

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
cython_files = ["pycylon/*/*.pyx"]

if not ARROW_HOME:
    raise ValueError("ARROW_HOME not set")

# For now, assume that we build against bundled pyarrow releases.

std_version = '-std=c++14'
pyarrow_include_dir = os.path.join(pyarrow_location, 'include')
extra_compile_args = os.popen("mpic++ --showme:compile").read().strip().split(' ')
extra_link_args = os.popen("mpic++ --showme:link").read().strip().split(' ')
additional_compile_args = [std_version,
                           '-DARROW_METADATA_V4 -DNEED_EXCLUSIVE_SCAN']
extra_compile_args = extra_link_args + additional_compile_args
extra_link_args.append("-Wl,-rpath,$ORIGIN/pyarrow")

arrow_library_directory = os.path.join(ARROW_HOME, "arrow", "install", "lib")
arrow_lib_include_dir = os.path.join(ARROW_HOME, "arrow", "install", "include")
cylon_library_directory = os.path.join(ARROW_HOME, "lib")

library_directories = [cylon_library_directory,
                       arrow_library_directory,
                       get_python_lib(),
                       os.path.join(os.sys.prefix, "lib")]

libraries = ["arrow", "cylon", "cylon_python"]

_include_dirs = ["../cpp/src/cylon/python",
                 "../cpp/src/cylon/lib",
                 "../cpp/src/cylon/",
                 "../cpp/src/cylon/net",
                 "../cpp/src/cylon/data",
                 "../cpp/src/cylon/io",
                 "../cpp/src/cylon/join",
                 "../cpp/src/cylon/util",
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
packages = find_packages(include=["pycylon", "pycylon.*"])

setup(
    name="pycylon",
    packages=packages,
    version='0.1.0',
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
