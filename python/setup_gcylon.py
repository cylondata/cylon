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
import sys
import numpy as np
from os.path import join as pjoin

import versioneer
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from distutils.sysconfig import get_python_lib

version = versioneer.get_version(),
cmdclass = versioneer.get_cmdclass(),

# make sure conda is activated or, conda-build is used
if ("CONDA_PREFIX" not in os.environ and "CONDA_BUILD" not in os.environ):
    print("Neither CONDA_PREFIX nor CONDA_BUILD is set. Activate conda environment or use conda-build")
    sys.exit()

def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    from: https://github.com/rmcgibbo/npcuda-example/blob/master/cython/setup.py
    """

    # First check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either add it to your path, '
                                   'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig

# os.environ["CXX"] = "mpic++"

class BuildExt(build_ext):
    def build_extensions(self):
        if '-Wstrict-prototypes' in self.compiler.compiler_so:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super().build_extensions()

try:
    nthreads = int(os.environ.get("PARALLEL_LEVEL", "0") or "0")
except Exception:
    nthreads = 0

std_version = '-std=c++17'
additional_compile_args = [std_version]

extra_compile_args = os.popen("mpic++ --showme:compile").read().strip().split(' ')
extra_link_args = os.popen("mpic++ --showme:link").read().strip().split(' ')
extra_compile_args = extra_compile_args + extra_link_args + additional_compile_args
#  extra_compile_args = additional_compile_args
# extra_link_args.append("-Wl,-rpath")

if "CONDA_BUILD" in os.environ:
    conda_lib_dir = os.path.join(os.environ.get('BUILD_PREFIX'), "lib") + " "
    conda_lib_dir += os.path.join(os.environ.get('PREFIX'), "lib")
    conda_include_dir = os.path.join(os.environ.get('BUILD_PREFIX'), "include") + " "
    conda_include_dir += os.path.join(os.environ.get('PREFIX'), "include")
elif "CONDA_PREFIX" in os.environ:
    conda_lib_dir = os.path.join(os.environ.get('CONDA_PREFIX'), "lib")
    conda_include_dir = os.path.join(os.environ.get('CONDA_PREFIX'), "include")

CUDA = locate_cuda()

print("conda_library_directory: ", conda_lib_dir)
print("cuda_library_directory: ", CUDA['lib64'])

library_directories = [
    conda_lib_dir,
    CUDA['lib64'],
    get_python_lib(),
    os.path.join(os.sys.prefix, "lib")]

libraries = ["gcylon", "cylon", "cudf", "cudart", "glog"]

_include_dirs = ["../cpp/src",
                 conda_include_dir,
                 os.path.join(conda_include_dir, "libcudf/libcudacxx"),
                 CUDA['include'],
                 np.get_include()]

cython_files = ["pygcylon/**/*.pyx"]

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

packages = find_packages(include=["pygcylon", "pygcylon.*"])

setup(
    name="pygcylon",
    packages=packages,
    version=version,
    setup_requires=["cython", "setuptools", "numpy"],
    ext_modules=cythonize(
        extensions,
        nthreads=nthreads,
        compiler_directives=dict(
            profile=False, language_level=3, embedsignature=True
        ),
    ),
    package_data=dict.fromkeys(
        find_packages(include=["pygcylon*"]), ["*.pxd"],
    ),
    python_requires='>=3.7',
    install_requires=["cython", "numpy", "cudf"],
    zip_safe=False,
)
