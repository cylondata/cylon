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

parser = argparse.ArgumentParser()

# C++ Build
cpp_build = parser.add_argument_group("Cylon C++")
cpp_build.add_argument("--cpp", action='store_true', help='Build C++ Core')
cpp_build.add_argument("--test", action='store_true',
                       help='Run C++ test suite')
cpp_build.add_argument("--style-check", action='store_true',
                       help='Compile C++ with style check')
cpp_build.add_argument("-spath", help='C++ source directory',
                       default=os.path.join(os.getcwd(), 'cpp'))
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
python_build.add_argument(
    "--cython", action='store_true', help='Build Cython modules')
python_build.add_argument("--python_with_pyarrow",
                          action='store_true', help='Build PyArrow with Python')
python_build.add_argument("-pyenv", help='Path to Python virtual environment',
                          default=os.path.join(os.getcwd(), 'ENV'))


# Java build
java_build = parser.add_argument_group("JCylon")
java_build.add_argument("--java", action='store_true',
                        help='Build JNI based Java wrapper')

# Docker build
java_build = parser.add_argument_group("Docker")
java_build.add_argument("--docker", action='store_true',
                        help='Build Cylon Docker images locally')

# Paths
parser.add_argument("-bpath", help='Build directory',
                    default=os.path.join(os.getcwd(), 'build'))
parser.add_argument("-ipath", help='Install directory')

args = parser.parse_args()

# Variables


def on_off(arg):
    return "ON" if arg else "OFF"


BUILD_CPP = args.cpp
CPP_BUILD_MODE = "Release" if (args.release or (
    not args.release and not args.debug)) else "Debug"
CPP_SOURCE_DIR = os.path.join(args.spath)
RUN_CPP_TESTS = args.test
CMAKE_FLAGS = args.cmake_flags
CPPLINT_COMMAND = " \"-DCMAKE_CXX_CPPLINT=cpplint;--linelength=100;--headers=h,hpp;--filter=-legal/copyright,-build/c++11,-runtime/references\" " if args.style_check else ""

# arrow build expects /s even on windows
PYTHON_ENV_PATH = os.path.join(args.pyenv)
BUILD_PYTHON = args.python

# docker
BUILD_DOCKER = args.docker

BUILD_DIR = os.path.join(args.bpath)
INSTALL_COMMAND = f"-DCMAKE_INSTALL_PREFIX={args.ipath}" if args.ipath else ""

# create build directory
if not os.path.exists(BUILD_DIR):
    print("Creating build directory...")
    os.makedirs(BUILD_DIR)


def env_activate_command():
    if os.name == 'posix':
        return os.path.join(PYTHON_ENV_PATH, 'bin', 'activate')
    elif os.name == 'nt':
        return os.path.join(PYTHON_ENV_PATH, 'Scripts', 'activate.bat')


def read_python_requirements():
    return subprocess.call(env_activate_command()+' && pip3 install -r requirements.txt || exit 1', shell=True)


if BUILD_PYTHON:
    read_python_requirements()


def print_line():
    print("=================================================================")


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

    print_line()
    print("Building CPP in", CPP_BUILD_MODE, "mode")
    print_line()

    cmake_command = f'cd {BUILD_DIR} && cmake -DPYCYLON_BUILD={on_off(BUILD_PYTHON)} -DPYTHON_EXEC_PATH=${PYTHON_ENV_PATH} \
      -DCMAKE_BUILD_TYPE={CPP_BUILD_MODE} -DCYLON_WITH_TEST={on_off(RUN_CPP_TESTS)} -DARROW_BUILD_TYPE=SYSTEM {CPPLINT_COMMAND} {INSTALL_COMMAND} \
      {CMAKE_FLAGS} \
      {CPP_SOURCE_DIR} || exit 1'

    if BUILD_PYTHON:
        cmake_command = env_activate_command()+' && '+cmake_command

    print(cmake_command)
    res = subprocess.call(cmake_command)
    check_status(res, "C++ cmake generate")

    cmake_build_command = f'cd {BUILD_DIR} && cmake --build . --parallel {os.cpu_count()}'
    if BUILD_PYTHON:
        cmake_build_command = env_activate_command()+' && '+cmake_build_command
    res = subprocess.call(cmake_build_command)
    check_status(res, "C++ cmake build")


def build_docker():
    if not BUILD_DOCKER:
        return


def python_test():
    test_command = env_activate_command(
    ) + f' && {python_command()} -m pytest python/test/test_all.py || exit 1'
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

    python_build_command = env_activate_command(
    ) + f' && pip3 uninstall -y pycylon && ${python_command()} python/setup.py install || exit 1'

    print(python_build_command)
    env = os.environ
    env["LD_LIBRARY_PATH"] = f'{os.path.join(BUILD_DIR, "arrow", "install","lib")}:{os.path.join(BUILD_DIR, "lib")}:{os.environ.get("LD_LIBRARY_PATH","")}'
    env["CYLON_PREFIX"] = BUILD_DIR
    env["ARROW_PREFIX"] = os.path.join(BUILD_DIR, "arrow", "install")

    res = subprocess.run(python_build_command, shell=True, env=env)

    check_status(res.returncode, "PyCylon build")


# build_cpp()
# build_docker()
# build_python()
# python_test()
print(python_command())
