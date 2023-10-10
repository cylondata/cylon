#!/usr/bin/env python

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
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

logging.basicConfig(format='[%(levelname)s] %(message)s')
logger = logging.getLogger("cylon_build")
logger.setLevel(logging.INFO)


def check_status(status, task):
    if status != 0:
        logger.error(f'{task} failed with a non zero exit code ({status}). '
                     f'Cylon build is terminating.')
        sys.exit(1)
    else:
        logger.info(f'{task} completed successfully')


def check_conda_prefix():
    conda_prefix = os.getenv('CONDA_PREFIX')
    if not conda_prefix:
        logger.error("The build should be in a conda environment")
        sys.exit(1)

    return conda_prefix


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
    "-cmake-flags", "--cmake-flags", help='Additional cmake flags', default='')

# Build mode
build_mode = parser.add_argument_group("Build Mode").add_mutually_exclusive_group()
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

# Java build
java_build = parser.add_argument_group("JCylon")
java_build.add_argument("--java", action='store_true',
                        help='Build JCylon')

# Docker build
docker_build = parser.add_argument_group("Docker")
docker_build.add_argument("--docker", action='store_true',
                          help='Build Cylon Docker images locally')

# Paths
parser.add_argument("-bpath", help='Build directory',
                    default=Path(os.getcwd(), 'build'))
parser.add_argument("-ipath", "--prefix", dest='ipath', help='Install directory')

parser.add_argument("--verbose", help='Set verbosity', default=False, action="store_true")
parser.add_argument("-j", help='Parallel build threads', default=os.cpu_count(),
                    dest='parallel', type=int)
parser.add_argument("--clean", action='store_true', help="Clean before building")

args = parser.parse_args()


# Variables
def on_off(arg):
    return "ON" if arg else "OFF"


BUILD_CPP = args.cpp
CPP_BUILD_MODE = "Release" if (args.release or (
        not args.release and not args.debug)) else "Debug"
CPP_SOURCE_DIR = str(Path(args.root, 'cpp'))
PYTHON_SOURCE_DIR = Path(args.root, 'python', 'pycylon')
JAVA_SOURCE_DIR = Path(args.root, 'java')
RUN_CPP_TESTS = args.test
RUN_PYTHON_TESTS = args.pytest
CMAKE_FLAGS = args.cmake_flags
PARALLEL = args.parallel

# arrow build expects /s even on windows
BUILD_PYTHON = args.python

# java
BUILD_JAVA = args.java

# docker
BUILD_DOCKER = args.docker

BUILD_DIR = str(Path(args.bpath))
INSTALL_DIR = str(Path(args.ipath or check_conda_prefix()))

OS_NAME = platform.system()  # Linux, Darwin or Windows

PYTHON_EXEC = sys.executable

CPPLINT_COMMAND = ""


def check_and_install_cpplint():
    global CPPLINT_COMMAND
    if args.style_check:
        cmd = f'{PYTHON_EXEC} -m pip install cpplint'
        res = subprocess.run(cmd, shell=True, cwd=PYTHON_SOURCE_DIR)
        check_status(res.returncode, "cpplint install")

        CPPLINT_COMMAND = "-DCMAKE_CXX_CPPLINT=\"cpplint;--linelength=100;--headers=h," \
                          "hpp;--filter=-legal/copyright,-build/c++11,-runtime/references\" "


CMAKE_BOOL_FLAGS = {'CYLON_GLOO', 'CYLON_UCX', 'CYLON_UCC'}
CMAKE_FALSE_OPTIONS = {'0', 'FALSE', 'OFF', 'N', 'NO', 'IGNORE', 'NOTFOUND'}


def parse_cmake_bool(v):
    """
    Converts string to 0 or 1. Evaluates to 0 if any of the following is true:
    - string is empty,
    - string is a case-insensitive equal of 0, FALSE, OFF, N, NO, IGNORE, or NOTFOUND, or
    - string ends in the suffix -NOTFOUND (case-sensitive).
    Otherwise, evaluates to 1.
    """
    return bool(v and v not in CMAKE_FALSE_OPTIONS)


def parse_cmake_flags(flag):
    for f in CMAKE_FLAGS.strip().split('-D'):
        if not f.strip():
            continue
        strip = f.strip()

        k, v = f.strip().split('=')
        if k != flag:
            continue
        else:
            return parse_cmake_bool(v) if k in CMAKE_BOOL_FLAGS else v
    return None


CYLON_GLOO = parse_cmake_flags('CYLON_GLOO')
GLOO_PREFIX = parse_cmake_flags('GLOO_INSTALL_PREFIX')
CYLON_REDIS = parse_cmake_flags("CYLON_USE_REDIS")
CYLON_UCX = parse_cmake_flags('CYLON_UCX')
CYLON_UCC = parse_cmake_flags('CYLON_UCC')
UCX_INSTALL_PREFIX = parse_cmake_flags('UCX_INSTALL_PREFIX')
UCC_PREFIX = parse_cmake_flags('UCC_INSTALL_PREFIX')
REDIS_PREFIX = parse_cmake_flags('REDIS_INSTALL_PREFIX')

if CYLON_REDIS and (REDIS_PREFIX is None):
    REDIS_PREFIX = "/usr/local"


def print_line():
    logger.info("=================================================================")


print_line()
logger.info(f"OS             : {OS_NAME}")
logger.info(f"Python exec    : {PYTHON_EXEC}")
logger.info(f"Build mode     : {CPP_BUILD_MODE}")
logger.info(f"Build path     : {BUILD_DIR}")
logger.info(f"Build threads  : {PARALLEL}")
logger.info(f"Install path   : {INSTALL_DIR}")
logger.info(f"CMake flags    : {CMAKE_FLAGS}")
logger.info(f" -CYLON_GLOO   : {CYLON_GLOO}")
logger.info(f" -GLOO_PREFIX  : {GLOO_PREFIX}")
logger.info(f" -REDIS_PREFIX  : {REDIS_PREFIX}")
logger.info(f" -CYLON_UCX    : {CYLON_UCX}")
logger.info(f" -CYLON_UCC    : {CYLON_UCC}")
logger.info(f" -UCC_PREFIX   : {UCC_PREFIX}")
logger.info(f"Run C++ tests  : {RUN_CPP_TESTS}")
logger.info(f"Build PyCylon  : {BUILD_PYTHON}")
logger.info(f"Run Py tests   : {RUN_PYTHON_TESTS}")
print_line()

# create build directory
if not os.path.exists(BUILD_DIR):
    logger.info("Creating build directory...")
    os.makedirs(BUILD_DIR)


def build_cpp():
    if not BUILD_CPP:
        return

    win_cmake_args = "-A x64" if os.name == 'nt' else ""
    verb = '-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON' if args.verbose else ''
    clean = '--clean-first' if args.clean else ''

    cmake_command = f"cmake -DPYCYLON_BUILD={on_off(BUILD_PYTHON)} {win_cmake_args} " \
                    f"-DCMAKE_BUILD_TYPE={CPP_BUILD_MODE} " \
                    f"-DCYLON_WITH_TEST={on_off(RUN_CPP_TESTS)} " \
                    f"-DARROW_BUILD_TYPE=SYSTEM " \
                    f"{CPPLINT_COMMAND} " \
                    f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR} " \
                    f"{verb} {CMAKE_FLAGS} -permissive {CPP_SOURCE_DIR}"

    logger.info(f"Generate command: {cmake_command}")
    res = subprocess.call(cmake_command, cwd=BUILD_DIR, shell=True)
    check_status(res, "C++ cmake generate")

    cmake_build_command = f'cmake --build . --parallel {PARALLEL} --config {CPP_BUILD_MODE} {clean}'
    logger.info(f"Build command: {cmake_build_command}")
    res = subprocess.call(cmake_build_command, cwd=BUILD_DIR, shell=True)
    check_status(res, "C++ cmake build")

    cmake_install_command = f'cmake --install . --prefix {INSTALL_DIR}'
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
    logger.error("Docker build not implemented in this script")
    sys.exit(1)


def python_test():
    if not RUN_PYTHON_TESTS:
        return

    check_conda_prefix()

    env = os.environ
    if args.ipath:
        if OS_NAME == 'Linux':
            if 'LD_LIBRARY_PATH' in env:
                env['LD_LIBRARY_PATH'] = str(Path(INSTALL_DIR, "lib")) + os.pathsep \
                                         + env['LD_LIBRARY_PATH']
            else:
                env['LD_LIBRARY_PATH'] = str(Path(INSTALL_DIR, "lib"))
            logger.info(f"LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")

            if CYLON_GLOO:
                env['CYLON_GLOO'] = str(CYLON_GLOO)
                env['GLOO_PREFIX'] = GLOO_PREFIX

                env['LD_LIBRARY_PATH'] = os.path.join(GLOO_PREFIX, "lib") + os.pathsep + \
                                         env['LD_LIBRARY_PATH']

            if UCX_INSTALL_PREFIX is not None:
                env['UCX_LOCAL_INSTALL'] = '1'
            else:
                env['UCX_LOCAL_INSTALL'] = '0'

            if UCX_INSTALL_PREFIX is not None:
                env['UCX_INSTALL_PREFIX'] = UCX_INSTALL_PREFIX
                env['LD_LIBRARY_PATH'] = os.path.join(UCX_INSTALL_PREFIX, "lib") + os.pathsep + \
                                         os.path.join(UCX_INSTALL_PREFIX, "lib", "uct") + os.pathsep + \
                                         os.path.join(UCX_INSTALL_PREFIX, "lib", "ucs") + os.pathsep + \
                                         os.path.join(UCX_INSTALL_PREFIX, "lib", "ucm") + os.pathsep + \
                                         os.path.join(UCX_INSTALL_PREFIX, "lib", "ucp") + os.pathsep + \
                                         env['LD_LIBRARY_PATH']

            if CYLON_UCC:
                env['CYLON_UCC'] = str(CYLON_UCC)
                env['UCC_PREFIX'] = UCC_PREFIX
                env['LD_LIBRARY_PATH'] = os.path.join(UCC_PREFIX, "lib") + os.pathsep + \
                                         os.path.join(UCC_PREFIX, "lib", "ucc") + os.pathsep + \
                                         env['LD_LIBRARY_PATH']


            if CYLON_REDIS:
                env['CYLON_REDIS'] = str(CYLON_REDIS)
                env['REDIS_PREFIX'] = REDIS_PREFIX
                env['LD_LIBRARY_PATH'] = os.path.join(REDIS_PREFIX, "lib") + os.pathsep + \
                                         os.path.join(REDIS_PREFIX, "lib", "redis++") + os.pathsep + \
                                         os.path.join(REDIS_PREFIX, "lib", "hiredis") + os.pathsep + \
                                         env['LD_LIBRARY_PATH']

        elif OS_NAME == 'Darwin':
            if 'DYLD_LIBRARY_PATH' in env:
                env['DYLD_LIBRARY_PATH'] = str(Path(INSTALL_DIR, "lib")) + os.pathsep \
                                           + env['DYLD_LIBRARY_PATH']
            else:
                env['DYLD_LIBRARY_PATH'] = str(Path(INSTALL_DIR, "lib"))
            logger.info(f"DYLD_LIBRARY_PATH: {env['DYLD_LIBRARY_PATH']}")
        else:  # Windows
            env['PATH'] = str(Path(INSTALL_DIR, "bin")) + os.pathsep + env['PATH']
            logger.info(f"PATH: {env['PATH']}")

    test_command = f"{PYTHON_EXEC} -m pytest -v python/pycylon/test/test_all.py"
    res = subprocess.run(test_command, env=env, shell=True)
    check_status(res.returncode, "Python test suite")


def build_python():
    if not BUILD_PYTHON:
        return

    print_line()
    logger.info("Building Python")

    conda_prefix = check_conda_prefix()

    env = os.environ
    env["CYLON_PREFIX"] = str(INSTALL_DIR)
    if os.name == 'posix':
        env["ARROW_PREFIX"] = str(Path(conda_prefix))
    elif os.name == 'nt':
        env["ARROW_PREFIX"] = str(Path(os.environ["CONDA_PREFIX"], "Library"))

    if CYLON_GLOO:
        env['CYLON_GLOO'] = str(CYLON_GLOO)
        env['GLOO_PREFIX'] = GLOO_PREFIX
    if CYLON_UCC and CYLON_UCX:
        env['CYLON_UCX'] = str(CYLON_UCX)
        env['CYLON_UCC'] = str(CYLON_UCC)
        env['UCC_PREFIX'] = UCC_PREFIX

    if UCX_INSTALL_PREFIX is not None:
        env['UCX_LOCAL_INSTALL'] = '1'
        env['UCX_INSTALL_PREFIX'] = UCX_INSTALL_PREFIX
    else:
        env['UCX_LOCAL_INSTALL'] = '0'


    if CYLON_REDIS:
        env['CYLON_REDIS'] = str(CYLON_REDIS)
        env['REDIS_PREFIX'] = REDIS_PREFIX

    logger.info("Arrow prefix: " + str(Path(conda_prefix)))
    clean = '--upgrade' if args.clean else ''
    cmd = f'{PYTHON_EXEC} -m pip install -v {clean} .'
    res = subprocess.run(cmd, shell=True, env=env, cwd=PYTHON_SOURCE_DIR)
    check_status(res.returncode, "PyCylon build")


def build_java():
    if not BUILD_JAVA:
        return

    conda_prefix = check_conda_prefix()

    mvn_cmd = f"mvn clean install -Dcylon.core.libs={INSTALL_DIR}/lib " \
              f"-Dcylon.arrow.dir={conda_prefix}"
    res = subprocess.run(mvn_cmd, shell=True, cwd=JAVA_SOURCE_DIR)
    check_status(res.returncode, "JCylon build")


check_and_install_cpplint()
build_cpp()
build_python()
python_test()
build_java()
