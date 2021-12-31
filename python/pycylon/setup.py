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

import versioneer
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension
import glob

version = versioneer.get_version(),
cmdclass = versioneer.get_cmdclass(),

# os.environ["CXX"] = "mpic++"
pyarrow_location = os.path.dirname(pa.__file__)
pyarrow_version = pa.__version__

print("PYARROW version:", pyarrow_version)

CYLON_PREFIX = os.environ.get('CYLON_PREFIX')
ARROW_PREFIX = os.environ.get('ARROW_PREFIX')

try:
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0") or "0")
except Exception:
    nthreads = 0

compiler_directives = {"language_level": 3, "embedsignature": True}


cython_files = ["pycylon/*/*.pyx"]
print("CYTHON: " + str(cython_files))

if not CYLON_PREFIX:
    raise ValueError("CYLON_PREFIX not set")

std_version = '-std=c++14'
additional_compile_args = [std_version,
                           '-DARROW_METADATA_V4 -DNEED_EXCLUSIVE_SCAN']
arrow_lib_include_dir = None
arrow_library_directory = None
if not ARROW_PREFIX:
    arrow_lib_include_dir = os.path.join(pyarrow_location, "include")
    arrow_library_directory = pyarrow_location
    additional_compile_args = additional_compile_args + \
        ['-D_GLIBCXX_USE_CXX11_ABI=0']
    if not os.path.exists(arrow_library_directory):
        arrow_library_directory = os.path.join(pyarrow_location, "lib64")
else:
    arrow_lib_include_dir = os.path.join(ARROW_PREFIX, "include")
    arrow_library_directory = os.path.join(ARROW_PREFIX, "lib")
    if not os.path.exists(arrow_library_directory):
        arrow_library_directory = os.path.join(ARROW_PREFIX, "lib64")

pyarrow_include_dir = os.path.join(pyarrow_location, 'include')

extra_compile_args = []
extra_link_args = []
if os.name == 'posix':
    extra_compile_args = os.popen(
        "mpic++ --showme:compile").read().strip().split(' ')
    extra_link_args = os.popen("mpic++ --showme:link").read().strip().split(' ')
    extra_compile_args = extra_compile_args + additional_compile_args
    extra_link_args = ["-W"]

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

print("Libraries: " + str(library_directories))

libraries = ["arrow", "cylon", "glog", "mpi"] # todo glogd was added temporarily
cylon_include_dir = os.path.abspath(os.path.join(__file__, "../../..", "cpp", "src"))

_include_dirs = [cylon_include_dir,
                 arrow_lib_include_dir,
                 glog_lib_include_dir,
                 pyarrow_include_dir,
                 np.get_include(),
                 os.path.dirname(sysconfig.get_path("include")),
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

print("PACKAGES: " + str(packages))

ret = setup(
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
    ),
    package_data=dict.fromkeys(
        find_packages(include=["pycylon*"]), ["*.pxd"],
    ),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        f'pyarrow=={pyarrow_version}',
        'cython',
    ],
    zip_safe=False
)
print("Done setup ####################################")
