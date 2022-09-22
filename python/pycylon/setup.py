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
"""
https://github.com/FedericoStra/cython-package-example/blob/master/setup.py
https://github.com/thewtex/cython-cmake-example/blob/master/setup.py
"""

import os
import platform
import sys
import sysconfig
import traceback
from distutils.sysconfig import get_python_lib
from distutils.util import strtobool

import numpy as np
import pyarrow as pa
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

import versioneer

version = versioneer.get_version(),
cmdclass = versioneer.get_cmdclass(),

pyarrow_location = os.path.dirname(pa.__file__)
pyarrow_version = pa.__version__

print("PYARROW version:", pyarrow_version)

CYLON_PREFIX = os.environ.get('CYLON_PREFIX')
ARROW_PREFIX = os.environ.get('ARROW_PREFIX')
CYLON_GLOO = strtobool(os.environ.get('CYLON_GLOO') or '0')
GLOO_PREFIX = os.environ.get('GLOO_PREFIX')
CYLON_UCX = strtobool(os.environ.get('CYLON_UCX') or '0')
CYLON_UCC = strtobool(os.environ.get('CYLON_UCC') or '0')
UCC_PREFIX = os.environ.get('UCC_PREFIX')

if not CYLON_PREFIX:
    raise ValueError("CYLON_PREFIX not set")

try:
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0") or "0")
except Exception:
    nthreads = 0

compiler_directives = {"language_level": 3, "embedsignature": True}

std_version = '-std=c++14'
additional_compile_args = [std_version, '-DARROW_METADATA_V4 -DNEED_EXCLUSIVE_SCAN']
arrow_lib_include_dir = None
arrow_library_directory = None
if not ARROW_PREFIX:
    arrow_lib_include_dir = os.path.join(pyarrow_location, "include")
    arrow_library_directory = pyarrow_location
    additional_compile_args.append('-D_GLIBCXX_USE_CXX11_ABI=0')
    additional_compile_args.append('-DOMPI_SKIP_MPICXX=1')
    if not os.path.exists(arrow_library_directory):
        arrow_library_directory = os.path.join(pyarrow_location, "lib64")
else:
    arrow_lib_include_dir = os.path.join(ARROW_PREFIX, "include")
    arrow_library_directory = os.path.join(ARROW_PREFIX, "lib")
    if not os.path.exists(arrow_library_directory):
        arrow_library_directory = os.path.join(ARROW_PREFIX, "lib64")

pyarrow_include_dir = os.path.join(pyarrow_location, 'include')

extra_link_args = []
if os.name == 'posix':
    try:
        res_str = os.popen("mpicc -show").read().strip().split()
        for s in res_str:
            if s.startswith('-I', 0, 2):
                additional_compile_args.append(s[2:])
            if s.startswith('-L', 0, 2):
                extra_link_args.append(s[2:])
        extra_link_args.append("-W")
    except Exception:
        traceback.print_exception(*sys.exc_info())
        exit(1)

glob_library_directory = os.path.join(CYLON_PREFIX, "glog", "install", "lib")

glog_lib_include_dir = os.path.join(CYLON_PREFIX, "glog", "install", "include")
cylon_library_directory = os.path.join(CYLON_PREFIX, "lib")
cylon_library_directory_debug = os.path.join(CYLON_PREFIX, "lib", "Debug")
cylon_library_directory_release = os.path.join(CYLON_PREFIX, "lib", "Release")

library_directories = [cylon_library_directory,
                       cylon_library_directory_debug,
                       cylon_library_directory_release,
                       arrow_library_directory,
                       glob_library_directory,
                       get_python_lib(),
                       os.path.join(os.sys.prefix, "lib")]

OS_NAME = platform.system()

# add LD_LIBRARY_PATH libs to library_dirs
if OS_NAME == 'Linux' and os.environ.get('LD_LIBRARY_PATH'):
    library_directories.extend(os.environ['LD_LIBRARY_PATH'].split(':'))
elif OS_NAME == 'Darwin' and os.environ.get('DYLD_LIBRARY_PATH'):
    library_directories.extend(os.environ['DYLD_LIBRARY_PATH'].split(':'))

if OS_NAME == 'Linux' or OS_NAME == 'Darwin':
    mpi_library_dir = os.popen("mpicc --showme:libdirs").read().strip().split(' ')
else:
    import mpi4py

    mpi_library_dir = [mpi4py.get_config()['library_dirs']]
library_directories.extend(mpi_library_dir)

libraries = ["arrow", "cylon", "glog"]

cylon_include_dir = os.path.abspath(os.path.join(__file__, "../../..", "cpp", "src"))

_include_dirs = [cylon_include_dir,
                 arrow_lib_include_dir,
                 glog_lib_include_dir,
                 pyarrow_include_dir,
                 np.get_include(),
                 os.path.dirname(sysconfig.get_path("include"))]

if OS_NAME == 'Linux' or OS_NAME == 'Darwin':
    mpi_include_dir = os.popen("mpicc --showme:incdirs").read().strip().split(' ')
else:
    import mpi4py

    mpi_include_dir = [mpi4py.get_config()['include_dirs']]
_include_dirs.extend(mpi_include_dir)

macros = []
# compile_time_env serves as preprocessor macros. ref: https://github.com/cython/cython/issues/2488
compile_time_env = {'CYTHON_GLOO': False, 'CYTHON_UCC': False, 'CYTHON_UCX': False}
if CYLON_GLOO:
    libraries.append('gloo')
    library_directories.append(os.path.join(GLOO_PREFIX, 'lib'))
    _include_dirs.append(os.path.join(GLOO_PREFIX, 'include'))
    macros.append(('GLOO_USE_MPI', '1'))
    macros.append(('BUILD_CYLON_GLOO', '1'))
    compile_time_env['CYTHON_GLOO'] = True

if CYLON_UCC and CYLON_UCX:
    libraries.append('ucc')
    library_directories.append(os.path.join(UCC_PREFIX, 'lib'))
    _include_dirs.append(os.path.join(UCC_PREFIX, 'include'))
    macros.append(('BUILD_CYLON_UCX', '1'))
    macros.append(('BUILD_CYLON_UCC', '1'))
    compile_time_env['CYTHON_UCX'] = True
    compile_time_env['CYTHON_UCC'] = True

print('Libraries    :', libraries)
print("Lib dirs     :", library_directories)
print("Include dirs :", _include_dirs)
print("Macros       :", macros)
print("Compile time env:", compile_time_env)

# Adopted the Cudf Python Build format
# https://github.com/rapidsai/cudf

extensions = [
    Extension(
        "*",
        sources=["pycylon/*/*.pyx"],
        include_dirs=_include_dirs,
        language='c++',
        extra_compile_args=additional_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        library_dirs=library_directories,
        define_macros=macros,
    )]

compiler_directives = {"language_level": 3, "embedsignature": True}
packages = find_packages(include=["pycylon", "pycylon.*"])

print("PACKAGES: " + str(packages))

setup(
    name="pycylon",
    packages=packages,
    version=versioneer.get_version(),
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
        compile_time_env=compile_time_env,
    ),
    package_data=dict.fromkeys(find_packages(include=["pycylon*"]), ["*.pxd"], ),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        f'pyarrow=={pyarrow_version}',
        'cython',
    ],
    zip_safe=False
)
print("Pycylon setup done!")
