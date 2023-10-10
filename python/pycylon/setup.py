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

pyarrow_location = os.path.dirname(pa.__file__)
pyarrow_version = pa.__version__

CYLON_PREFIX = os.environ.get('CYLON_PREFIX')
ARROW_PREFIX = os.environ.get('ARROW_PREFIX')
CYLON_GLOO = strtobool(os.environ.get('CYLON_GLOO') or '0')
GLOO_PREFIX = os.environ.get('GLOO_PREFIX')
CYLON_UCX = strtobool(os.environ.get('CYLON_UCX') or '0')
CYLON_UCC = strtobool(os.environ.get('CYLON_UCC') or '0')
CYLON_REDIS = strtobool(os.environ.get('CYLON_REDIS') or '0')
UCX_LOCAL_INSTALL = strtobool(os.environ.get('UCX_LOCAL_INSTALL') or '0')
UCC_PREFIX = os.environ.get('UCC_PREFIX')

REDIS_PREFIX = os.environ.get('REDIS_PREFIX')


print("Cylon prefix:", CYLON_PREFIX)
print("Arrow prefix:", ARROW_PREFIX)
print("Arrow version:", pyarrow_version)
print("UCC prefix:", UCC_PREFIX)
print("CYLON REDIS: ", CYLON_REDIS)
print("REDIS prefix:", REDIS_PREFIX)




OS_NAME = platform.system()

if not CYLON_PREFIX:
    raise ValueError("CYLON_PREFIX not set")

try:
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0") or "0")
except Exception:
    nthreads = 0

include_dirs = []
library_dirs = []
libraries = []
extra_compile_args = []
extra_link_args = []

std_version = '-std=c++14'
extra_compile_args.extend([std_version, '-DARROW_METADATA_V4 -DNEED_EXCLUSIVE_SCAN'])
extra_compile_args.append('-DOMPI_SKIP_MPICXX=1')

arrow_include_dir = None
arrow_lib_dir = None
if not ARROW_PREFIX:
    arrow_include_dir = os.path.join(pyarrow_location, "include")
    arrow_lib_dir = pyarrow_location
    if not os.path.exists(arrow_lib_dir):
        arrow_lib_dir = os.path.join(pyarrow_location, "lib64")
    extra_compile_args.append('-D_GLIBCXX_USE_CXX11_ABI=0')
else:
    arrow_include_dir = os.path.join(ARROW_PREFIX, "include")
    arrow_lib_dir = os.path.join(ARROW_PREFIX, "lib")
    if not os.path.exists(arrow_lib_dir):
        arrow_lib_dir = os.path.join(ARROW_PREFIX, "lib64")
pyarrow_include_dir = os.path.join(pyarrow_location, 'include')

glog_lib_dir = os.path.join(CYLON_PREFIX, "glog", "install", "lib")
glog_include_dir = os.path.join(CYLON_PREFIX, "glog", "install", "include")

cylon_library_directory = os.path.join(CYLON_PREFIX, "lib")
cylon_library_directory_debug = os.path.join(CYLON_PREFIX, "lib", "Debug")
cylon_library_directory_release = os.path.join(CYLON_PREFIX, "lib", "Release")

cylon_include_dir = os.path.abspath(os.path.join(CYLON_PREFIX, "include"))

# add lib_dirs
library_dirs.extend([cylon_library_directory,
                     cylon_library_directory_debug,
                     cylon_library_directory_release,
                     arrow_lib_dir,
                     glog_lib_dir,
                     get_python_lib(),
                     os.path.join(sys.prefix, "lib")])

# add LD_LIBRARY_PATH libs to library_dirs
if OS_NAME == 'Linux' and os.environ.get('LD_LIBRARY_PATH'):
    library_dirs.extend(os.environ['LD_LIBRARY_PATH'].split(':'))
elif OS_NAME == 'Darwin' and os.environ.get('DYLD_LIBRARY_PATH'):
    library_dirs.extend(os.environ['DYLD_LIBRARY_PATH'].split(':'))

# add libraries
libraries.extend(["arrow", "cylon", "glog"])

# add include dirs
include_dirs.extend([cylon_include_dir,
                     arrow_include_dir,
                     glog_include_dir,
                     pyarrow_include_dir,
                     np.get_include(),
                     os.path.dirname(sysconfig.get_path("include"))])

# resolve MPI lib_dir and include_dir
if OS_NAME == 'Linux' or OS_NAME == 'Darwin':
    try:
        res_str = os.popen("mpicc -show").read().strip().split()
        for s in res_str:
            if s.startswith('-I', 0, 2):
                include_dirs.append(s[2:])
            elif s.startswith('-L', 0, 2):
                library_dirs.append(s[2:])
            elif s.startswith('-l', 0, 2):
                libraries.append(s[2:])
            elif s.startswith('-Wl', 0, 3):
                extra_link_args.append(s)
    except Exception:
        traceback.print_exception(*sys.exc_info())
        exit(1)
else:
    import mpi4py

    library_dirs.append(mpi4py.get_config()['library_dirs'])
    include_dirs.append(mpi4py.get_config()['include_dirs'])

macros = []
# compile_time_env serves as preprocessor macros. ref: https://github.com/cython/cython/issues/2488
compile_time_env = {'CYTHON_GLOO': False, 'CYTHON_UCC': False, 'CYTHON_UCX': False, 'CYTHON_REDIS': False}
if CYLON_GLOO:
    libraries.append('gloo')
    library_dirs.append(os.path.join(GLOO_PREFIX, 'lib'))
    include_dirs.append(os.path.join(GLOO_PREFIX, 'include'))
    macros.append(('GLOO_USE_MPI', '1'))
    macros.append(('BUILD_CYLON_GLOO', '1'))
    compile_time_env['CYTHON_GLOO'] = True
else:
    macros.append(('GLOO_USE_MPI', '0'))
    macros.append(('BUILD_CYLON_GLOO', '0'))

if UCX_LOCAL_INSTALL:
    print("UCX Local install")
    UCX_INSTALL_PREFIX = os.environ.get('UCX_INSTALL_PREFIX')
    libraries.append('uct')
    libraries.append('ucs')
    libraries.append('ucm')
    libraries.append('ucp')
    include_dirs.append(os.path.join(UCX_INSTALL_PREFIX, 'include'))
    library_dirs.append(os.path.join(UCX_INSTALL_PREFIX, 'lib'))

if CYLON_UCC and CYLON_UCX:
    libraries.append('ucc')
    library_dirs.append(os.path.join(UCC_PREFIX, 'lib'))
    include_dirs.append(os.path.join(UCC_PREFIX, 'include'))
    macros.append(('BUILD_CYLON_UCX', '1'))
    macros.append(('BUILD_CYLON_UCC', '1'))
    compile_time_env['CYTHON_UCX'] = True
    compile_time_env['CYTHON_UCC'] = True


else:
    macros.append(('BUILD_CYLON_UCX', '0'))
    macros.append(('BUILD_CYLON_UCC', '0'))

if CYLON_REDIS:
    libraries.append('hiredis')
    libraries.append('redis++')
    macros.append(('BUILD_CYLON_REDIS', '1'))
    compile_time_env['CYTHON_REDIS'] = True
    library_dirs.append(os.path.join(REDIS_PREFIX, 'lib'))
    library_dirs.append(os.path.join(REDIS_PREFIX, 'lib64'))
    include_dirs.append(os.path.join(REDIS_PREFIX, 'include', 'sw'))
    include_dirs.append(os.path.join(REDIS_PREFIX, 'include', 'hiredis'))
else:
    macros.append(('BUILD_CYLON_REDIS', '0'))


print('Libraries    :', libraries)
print("Lib dirs     :", library_dirs)
print("Include dirs :", include_dirs)
print("extra_compile_args :", extra_compile_args)
print("extra_link_args :", extra_link_args)
print("Macros       :", macros)
print("Compile time env:", compile_time_env)

# Adopted the Cudf Python Build format
# https://github.com/rapidsai/cudf

extensions = [
    Extension(
        "*",
        sources=["pycylon/*/*.pyx"],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        library_dirs=library_dirs,
        define_macros=macros,
    )]

compiler_directives = {"profile": False, "language_level": 3, "embedsignature": True}
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
        compiler_directives=compiler_directives,
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
