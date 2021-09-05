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

import argparse
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()

# C++ Build
cpp_build = parser.add_argument_group("Cylon C++")
cpp_build.add_argument("--cpp", action='store_true', help='Build C++ Core')
cpp_build.add_argument("--test", action='store_true',
                       help='Run C++ test suite')
cpp_build.add_argument("--style-check", action='store_true',
                       help='Compile C++ with style check')
cpp_build.add_argument("-root", help='Cylon Source root directory',
                       default=Path(os.getcwd()))
cpp_build.add_argument(
    "-cmake-flags", help='Additional cmake flags', default='')

# Build mode
build_mode = parser.add_argument_group(
    "Build Mode").add_mutually_exclusive_group()
build_mode.add_argument("--debug", action='store_true',
                        help='Build the core in debug mode')
build_mode.add_argument("--release", action='store_true',
                        help='Build the core in release mode')

# Python build
python_build = parser.add_argument_group("PyCylon")
python_build.add_argument(
    "--python", action='store_true', help='Build PyCylon wrapper')
python_build.add_argument(
    "--pytest", action='store_true', help='Run Python test suite')

# Docker build
java_build = parser.add_argument_group("Docker")
java_build.add_argument("--docker", action='store_true',
                        help='Build Cylon Docker images locally')

# Paths
parser.add_argument("-bpath", help='Build directory',
                    default=Path(os.getcwd(), 'build'))
parser.add_argument("-ipath", help='Install directory')

args = parser.parse_args()

# Variables


def on_off(arg):
    return "ON" if arg else "OFF"


BUILD_CPP = args.cpp
CPP_BUILD_MODE = "Release" if (args.release or (
    not args.release and not args.debug)) else "Debug"
CPP_SOURCE_DIR = str(Path(args.root, 'cpp'))
PYTHON_SOURCE_DIR = Path(args.root, 'python', 'pycylon')
RUN_CPP_TESTS = args.test
CMAKE_FLAGS = args.cmake_flags
CPPLINT_COMMAND = " \"-DCMAKE_CXX_CPPLINT=cpplint;--linelength=100;--headers=h,hpp;--filter=-legal/copyright,-build/c++11,-runtime/references\" " if args.style_check else ""

# arrow build expects /s even on windows
BUILD_PYTHON = args.python

# docker
BUILD_DOCKER = args.docker

BUILD_DIR = str(Path(args.bpath))
INSTALL_COMMAND = f"-DCMAKE_INSTALL_PREFIX={args.ipath}" if args.ipath else ""

def print_line():
    print("=================================================================")

print_line()
print("Build mode: " + CPP_BUILD_MODE)
print("Build path: " + BUILD_DIR)
print_line()

# create build directory
if not os.path.exists(BUILD_DIR):
    print("Creating build directory...")
    os.makedirs(BUILD_DIR)

def check_status(status, task):
    if status != 0:
        print(f'{task} failed with a non zero exit code. Cylon build is terminating.')
        quit()
    else:
        print(f'{task} completed successfully')

def python_command():
    res = subprocess.call("python3 --version")
    if res == 0:
        return "python3"

    res = subprocess.call("python --version")
    if res == 0:
        return "python"

    print("Python not found.")
    quit()


def build_cpp():
    if not BUILD_CPP:
        return

    CONDA_PREFIX = os.getenv('CONDA_PREFIX')
    INSTALL_COMMAND = f"-DCMAKE_INSTALL_PREFIX={args.ipath}" if args.ipath else ""
    if not args.ipath:
        if CONDA_PREFIX:
            INSTALL_COMMAND = f"-DCMAKE_INSTALL_PREFIX={CONDA_PREFIX}"
        else:
            print("The build should be in a conda environment")
            return

    cmake_command = f'cmake -DPYCYLON_BUILD={on_off(BUILD_PYTHON)}  \
      -DCMAKE_BUILD_TYPE={CPP_BUILD_MODE} -DCYLON_WITH_TEST={on_off(RUN_CPP_TESTS)} -DARROW_BUILD_TYPE=SYSTEM {CPPLINT_COMMAND} {INSTALL_COMMAND} \
      {CMAKE_FLAGS} {CPP_SOURCE_DIR}'

    print(cmake_command, BUILD_DIR)
    res = subprocess.call(cmake_command, cwd=BUILD_DIR, shell=True)
    check_status(res, "C++ cmake generate")

    cmake_build_command = f'cmake --build . --parallel {os.cpu_count()} --config {CPP_BUILD_MODE}'
    res = subprocess.call(cmake_build_command, cwd=BUILD_DIR, shell=True)
    check_status(res, "C++ cmake build")

    cmake_install_command = f'cmake --install . --prefix {CONDA_PREFIX}'
    res = subprocess.call(cmake_install_command, cwd=BUILD_DIR, shell=True)
    check_status(res, "C++ cmake build")

def build_docker():
    if not BUILD_DOCKER:
        return


def python_test():
    test_command = '{python_command()} -m pytest python/test/test_all.py || exit 1'
    # ARROW_LIB=$(python3 -c 'import pyarrow as pa; import os; print(os.path.dirname(pa.__file__))') || exit 1
    # export LD_LIBRARY_PATH="${ARROW_LIB}:${BUILD_PATH}/lib:${LD_LIBRARY_PATH}" || exit 1
    # echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    env = os.environ
    env["LD_LIBRARY_PATH"] = f'{os.path.join(BUILD_DIR, "arrow", "install","lib")}:{os.path.join(BUILD_DIR, "lib")}:{os.environ.get("LD_LIBRARY_PATH","")}'
    env["PATH"] = f'{os.path.join(BUILD_DIR, "arrow", "install","lib")}:{os.path.join(BUILD_DIR, "lib")}:{os.environ.get("PATH","")}'
    res = subprocess.run(test_command, shell=True, env=env)
    check_status(res, "Python test suite")


def build_python():
    if not BUILD_PYTHON:
        return

    print_line()
    print("Building Python")

    CONDA_PREFIX = os.getenv('CONDA_PREFIX')
    if not CONDA_PREFIX:
        print("The build should be in a conda environment")
        return

    python_build_command = 'python setup.py install'
    env = os.environ
    env["CYLON_PREFIX"] = str(BUILD_DIR)
    if os.name == 'posix':
        env["ARROW_PREFIX"] = str(Path(CONDA_PREFIX))
    elif os.name == 'nt':
        env["ARROW_PREFIX"] = str(Path(os.environ["CONDA_PREFIX"], "Library"))

    print("Arrow prefix: " + str(Path(os.environ["CONDA_PREFIX"])))
    res = subprocess.run(python_build_command, shell=True, env=env, cwd=PYTHON_SOURCE_DIR)
    check_status(res.returncode, "PyCylon build")

build_cpp()
build_python()
