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
import logging
import platform
from pathlib import Path

import sys

logging.basicConfig(format='[%(levelname)s] %(message)s')
logger = logging.getLogger("cylon_build")
logger.setLevel(logging.INFO)

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

parser.add_argument("--verbose", help='Set verbosity', default=False, action="store_true")

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
RUN_PYTHON_TESTS = args.pytest
CMAKE_FLAGS = args.cmake_flags
CPPLINT_COMMAND = "\"-DCMAKE_CXX_CPPLINT=cpplint;--linelength=100;--headers=h," \
                  "hpp;--filter=-legal/copyright,-build/c++11,-runtime/references\" " if \
    args.style_check else " "

# arrow build expects /s even on windows
BUILD_PYTHON = args.python

# docker
BUILD_DOCKER = args.docker

BUILD_DIR = str(Path(args.bpath))
INSTALL_DIR = str(Path(args.ipath)) if args.ipath else ""

OS_NAME = platform.system()  # Linux, Darwin or Windows

PYTHON_EXEC = sys.executable


def print_line():
    logger.info("=================================================================")


print_line()
logger.info(f"OS             : {OS_NAME}")
logger.info(f"Python exec    : {PYTHON_EXEC}")
logger.info(f"Build mode     : {CPP_BUILD_MODE}")
logger.info(f"Build path     : {BUILD_DIR}")
logger.info(f"Install path   : {INSTALL_DIR}")
logger.info(f"CMake flags    : {CMAKE_FLAGS}")
logger.info(f"Run C++ tests  : {RUN_CPP_TESTS}")
logger.info(f"Build PyCylon  : {BUILD_PYTHON}")
logger.info(f"Run Py tests   : {RUN_PYTHON_TESTS}")
print_line()

# create build directory
if not os.path.exists(BUILD_DIR):
    logger.info("Creating build directory...")
    os.makedirs(BUILD_DIR)


def check_status(status, task):
    if status != 0:
        logger.error(f'{task} failed with a non zero exit code ({status}). '
                     f'Cylon build is terminating.')
        quit()
    else:
        logger.info(f'{task} completed successfully')


def build_cpp():
    if not BUILD_CPP:
        return

    CONDA_PREFIX = os.getenv('CONDA_PREFIX')
    if args.ipath:
        install_prefix = INSTALL_DIR
    else:
        if CONDA_PREFIX:
            install_prefix = CONDA_PREFIX
        else:
            logger.error(
                "install prefix can not be inferred. The build should be in a conda environment")
            return

    win_cmake_args = "-A x64" if os.name == 'nt' else ""
    verb = '-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON' if args.verbose else ''

    cmake_command = f"cmake -DPYCYLON_BUILD={on_off(BUILD_PYTHON)} {win_cmake_args} " \
                    f"-DCMAKE_BUILD_TYPE={CPP_BUILD_MODE} " \
                    f"-DCYLON_WITH_TEST={on_off(RUN_CPP_TESTS)} " \
                    f"-DARROW_BUILD_TYPE=SYSTEM " \
                    f"{CPPLINT_COMMAND} " \
                    f"-DCMAKE_INSTALL_PREFIX={install_prefix} " \
                    f"{verb} {CMAKE_FLAGS} {CPP_SOURCE_DIR}"

    logger.info(f"Generate command: {cmake_command}")
    res = subprocess.call(cmake_command, cwd=BUILD_DIR, shell=True)
    check_status(res, "C++ cmake generate")

    cmake_build_command = f'cmake --build . --parallel {os.cpu_count()} --config {CPP_BUILD_MODE}'
    logger.info(f"Build command: {cmake_build_command}")
    res = subprocess.call(cmake_build_command, cwd=BUILD_DIR, shell=True)
    check_status(res, "C++ cmake build")

    cmake_install_command = f'cmake --install . --prefix {install_prefix}'
    logger.info(f"Install command: {cmake_install_command}")
    res = subprocess.call(cmake_install_command, cwd=BUILD_DIR, shell=True)
    check_status(res, "C++ cmake install")

    if RUN_CPP_TESTS:
        cmake_test_command = f'cmake --build . --target test --config {CPP_BUILD_MODE}'
        if os.name == 'nt':
            cmake_test_command = f'cmake --build . --target RUN_TESTS --config {CPP_BUILD_MODE}'
        logger.info("CPP Test command: " + cmake_test_command)
        res = subprocess.call(cmake_test_command, cwd=BUILD_DIR, shell=True)
        check_status(res, "C++ cmake test")


def build_docker():
    if not BUILD_DOCKER:
        return


def python_test():
    if not RUN_PYTHON_TESTS:
        return
    env = os.environ
    if args.ipath:
        if OS_NAME == 'Linux':
            if 'LD_LIBRARY_PATH' in env:
                env['LD_LIBRARY_PATH'] = str(Path(INSTALL_DIR, "lib")) + os.pathsep \
                                         + env['LD_LIBRARY_PATH']
            else:
                env['LD_LIBRARY_PATH'] = str(Path(INSTALL_DIR, "lib"))
            logger.info(f"LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
        elif OS_NAME == 'Darwin':
            if 'DYLD_LIBRARY_PATH' in env:
                env['DYLD_LIBRARY_PATH'] = str(Path(INSTALL_DIR, "lib")) + os.pathsep \
                                           + env['DYLD_LIBRARY_PATH']
            else:
                env['DYLD_LIBRARY_PATH'] = str(Path(INSTALL_DIR, "lib"))
            logger.info(f"DYLD_LIBRARY_PATH: {env['DYLD_LIBRARY_PATH']}")
        else:  # Windows
            env['PATH'] = str(Path(INSTALL_DIR, "Library")) + os.pathsep + env['PATH']
            logger.info(f"PATH: {env['PATH']}")

    test_command = f"{PYTHON_EXEC} -m pytest -v python/pycylon/test/test_all.py"
    res = subprocess.run(test_command, env=env, shell=True)
    check_status(res.returncode, "Python test suite")


def build_python():
    if not BUILD_PYTHON:
        return

    print_line()
    logger.info("Building Python")

    CONDA_PREFIX = os.getenv('CONDA_PREFIX')
    if not CONDA_PREFIX:
        logger.error("The build should be in a conda environment")
        return

    python_build_command = f'{PYTHON_EXEC} setup.py install --force'
    env = os.environ
    env["CYLON_PREFIX"] = str(BUILD_DIR)
    if os.name == 'posix':
        env["ARROW_PREFIX"] = str(Path(CONDA_PREFIX))
    elif os.name == 'nt':
        env["ARROW_PREFIX"] = str(Path(os.environ["CONDA_PREFIX"], "Library"))

    logger.info("Arrow prefix: " + str(Path(os.environ["CONDA_PREFIX"])))
    res = subprocess.run(python_build_command, shell=True, env=env, cwd=PYTHON_SOURCE_DIR)
    check_status(res.returncode, "PyCylon build")


build_cpp()
build_python()
python_test()
