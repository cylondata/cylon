#!/bin/bash
#
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
#
SOURCE_DIR=$(pwd)/cpp
CPP_BUILD="OFF"
PYTHON_BUILD="OFF"
PYTHON_WITH_PYARROW_BUILD="OFF"
CONDA_CPP_BUILD="OFF"
CONDA_PYTHON_BUILD="OFF"
CYTHON_BUILD="OFF"
JAVA_BUILD="OFF"
BUILD_MODE=Release
BUILD_MODE_DEBUG="OFF"
BUILD_MODE_RELEASE="OFF"
PYTHON_RELEASE="OFF"
RUN_CPP_TESTS="OFF"
RUN_PYTHON_TESTS="OFF"
RUN_PYGCYLON_TESTS="OFF"
STYLE_CHECK="OFF"
INSTALL_PATH=
BUILD_PATH=$(pwd)/build
CMAKE_FLAGS=""
GCYLON_BUILD="OFF"
PYGCYLON_BUILD="OFF"
MAKE_JOBS=4

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"


case $key in
    -pyenv|--python_env_path)
    PYTHON_ENV_PATH="$2"
    shift # past argument
    shift # past value
    ;;
    -bpath|--build_path)
    BUILD_PATH="$2"
    shift # past argument
    shift # past value
    ;;
    -ipath|--install_path)
    INSTALL_PATH="$2"
    shift # past argument
    shift # past value
    ;;
    --cpp)
    CPP_BUILD="ON"
    shift # past argument
    ;;
    --python)
    CPP_BUILD="ON"
    PYTHON_BUILD="ON"
    shift # past argument
    ;;
    --conda_cpp)
    CONDA_CPP_BUILD="ON"
    shift # past argument
    ;;
    --conda_python)
    CONDA_PYTHON_BUILD="ON"
    shift # past argument
    ;;
    --python_with_pyarrow)
    PYTHON_WITH_PYARROW_BUILD="ON"
    shift # past argument
    ;;
    --cython)
    CYTHON_BUILD="ON"
    shift # past argument
    ;;
    --java)
    CPP_BUILD="ON"
    JAVA_BUILD="ON"
    shift # past argument
    ;;
    --debug)
    BUILD_MODE_DEBUG="ON"
    BUILD_MODE_RELEASE="OFF"
    shift # past argument
    ;;
    --release)
    BUILD_MODE_RELEASE="ON"
    BUILD_MODE_DEBUG="OFF"
    shift # past argument
    ;;
    --test)
    RUN_CPP_TESTS="ON"
    shift # past argument
    ;;
    --pytest)
    RUN_PYTHON_TESTS="ON"
    shift # past argument
    ;;
    --pygtest)
    RUN_PYGCYLON_TESTS="ON"
    shift # past argument
    ;;
    --style-check)
    STYLE_CHECK="ON"
    shift # past argument
    ;;
    --py-release)
    PYTHON_RELEASE="ON"
    CPP_BUILD="OFF"
    shift # past argument
    ;;
    --cmake-flags)
    CMAKE_FLAGS="$2"
    shift # past argument
    shift # past value
    ;;
    --gcylon)
    GCYLON_BUILD="ON"
    shift # past argument
    ;;
    --pygcylon)
    PYGCYLON_BUILD="ON"
    shift # past argument
    ;;
    -j|--jobs)
    MAKE_JOBS="$2"
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo "PYTHON ENV PATH       = ${PYTHON_ENV_PATH}"
echo "BUILD PATH            = ${BUILD_PATH}"
echo "FLAG CPP BUILD        = ${CPP_BUILD}"
echo "FLAG PYTHON BUILD     = ${PYTHON_BUILD}"
echo "FLAG BUILD DEBUG      = ${BUILD_MODE_DEBUG}"
echo "FLAG BUILD RELEASE    = ${BUILD_MODE_RELEASE}"
echo "FLAG RUN CPP TEST     = ${RUN_CPP_TESTS}"
echo "FLAG RUN PYTHON TEST  = ${RUN_PYTHON_TESTS}"
echo "RUN PYGCYLON TESTS    = ${RUN_PYGCYLON_TESTS}"
echo "FLAG STYLE CHECK      = ${STYLE_CHECK}"
echo "ADDITIONAL CMAKE FLAGS= ${CMAKE_FLAGS}"
echo "MAKE MAKE_JOBS        = ${MAKE_JOBS}"

if [[ -n $1 ]]; then
    echo "Last line of file specified as non-opt/last argument:"
fi

CPPLINT_COMMAND=" \"-DCMAKE_CXX_CPPLINT=cpplint;--linelength=100;--headers=h,hpp;--filter=-legal/copyright,-build/c++11,-runtime/references\" "

################################## Util Functions ##################################################
quit() {
exit
}

print_line() {
echo "=================================================================";
}

read_python_requirements(){
  pip3 install -r requirements.txt || exit 1
}

check_python_pre_requisites(){
  echo "Checking Python Pre_-requisites"
  response=$(python3 -c \
    "import numpy; print('Numpy Installation');\
    print('Version {}'.format(numpy.__version__));\
    print('Library Installation Path {}'.format(numpy.get_include()))")
  echo "${response}"
}

export_library_path() {
  if [ "$(uname)" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH=${1} || exit 1
    echo "DYLD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
  elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    export LD_LIBRARY_PATH=${1} || exit 1
    echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
  fi
}

INSTALL_CMD=
if [ -z "$INSTALL_PATH" ]
then
  echo "\-ipath|--install_path is NOT set default to cmake"
else
  INSTALL_CMD="-DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}"
  echo "Install location set to: ${INSTALL_PATH}"
fi

install_libs() {
  if [ -z "$INSTALL_PATH" ]
  then
    echo "\-ipath|--install_path is NOT set. we are not trying to install"
  else
    make install
  fi
}

build_cpp(){
  print_line
  echo "Building CPP in ${BUILD_MODE} mode"
  print_line

  if [ "${GCYLON_BUILD}" = "ON" ]; then
    echo "gcylon build is supported only with conda environment: --conda_cpp --with-gcylon"
    exit 1
  fi

  if [ "${PYTHON_BUILD}" = "ON" ]; then
    echo "Using Python environment at [${PYTHON_ENV_PATH}]"
    source "${PYTHON_ENV_PATH}"/bin/activate || exit 1
    read_python_requirements
  fi
  CPPLINT_CMD=" "
  if [ "${STYLE_CHECK}" = "ON" ]; then
    CPPLINT_CMD=${CPPLINT_COMMAND}
  fi
  mkdir -p ${BUILD_PATH}
  pushd ${BUILD_PATH} || exit 1
  export ARROW_HOME=${BUILD_PATH}/arrow/install
  cmake -DPYCYLON_BUILD=${PYTHON_BUILD} -DCMAKE_BUILD_TYPE=${BUILD_MODE} \
        -DCYLON_WITH_TEST=${RUN_CPP_TESTS} $CPPLINT_CMD $INSTALL_CMD \
      ${CMAKE_FLAGS} \
      ${SOURCE_DIR} || exit 1
  make -j ${MAKE_JOBS} || exit 1
  install_libs
  printf "ARROW HOME SET :%s \n" "${ARROW_HOME}"
  printf "Cylon CPP Built Successfully!"
  popd || exit 1
  print_line
}

build_cpp_with_custom_arrow(){
  print_line
  echo "Building CPP in ${BUILD_MODE} mode"
  print_line
  source "${PYTHON_ENV_PATH}"/bin/activate || exit 1
  read_python_requirements
  ARROW_LIB=$(python3 -c 'import pyarrow as pa; import os; print(os.path.dirname(pa.__file__))') || exit 1
  ARROW_INC=$(python3 -c 'import pyarrow as pa; import os; print(os.path.join(os.path.dirname(pa.__file__), "include"))')  || exit 1
  echo "ARROW_LIB: $ARROW_LIB"
  echo "ARROW_INC: $ARROW_INC"

# sometimes pip pyarrow installation does not contain a libarrow.so file, but only libarrow.so.xxx.
# then, create a symlink libarrow.so -->  libarrow.so.xxx
  for SO_FILE in "${ARROW_LIB}/libarrow.so" "${ARROW_LIB}/libarrow_python.so"; do
  if [ ! -f "$SO_FILE" ]; then
    echo "$SO_FILE does not exist! Trying to create a symlink"
    ln -sf "$(ls "$SO_FILE".*)" "$SO_FILE" || exit 1
  fi
  done

  CPPLINT_CMD=" "
  if [ "${STYLE_CHECK}" = "ON" ]; then
    CPPLINT_CMD=${CPPLINT_COMMAND}
  fi
  echo "SOURCE_DIR: ${SOURCE_DIR}"
  mkdir -p ${BUILD_PATH}
  pushd ${BUILD_PATH} || exit 1
  cmake -DPYCYLON_BUILD=${PYTHON_BUILD} -DCMAKE_BUILD_TYPE=${BUILD_MODE} \
      -DCYLON_WITH_TEST=${RUN_CPP_TESTS} $CPPLINT_CMD $INSTALL_CMD \
      -DARROW_BUILD_TYPE="CUSTOM" -DARROW_LIB_DIR=${ARROW_LIB} -DARROW_INCLUDE_DIR=${ARROW_INC} \
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
      ${CMAKE_FLAGS} \
      ${SOURCE_DIR} \
      || exit 1
  make -j ${MAKE_JOBS} || exit 1
  install_libs
  printf "Cylon CPP Built Successfully!"
  popd || exit 1
  print_line
}

build_cpp_conda(){
  print_line
  echo "Building Conda CPP in ${BUILD_MODE} mode"
  print_line

  # set install path to conda directory if not already set
  INSTALL_PATH=${INSTALL_PATH:=${PREFIX:=${CONDA_PREFIX}}}

  ARROW_LIB=${CONDA_PREFIX}/lib
  ARROW_INC=${CONDA_PREFIX}/include
  echo "ARROW_LIB: $ARROW_LIB"
  echo "ARROW_INC: $ARROW_INC"

  # sometimes pip pyarrow installation does not contain a libarrow.so file, but only libarrow.so.xxx.
  # then, create a symlink libarrow.so -->  libarrow.so.xxx
  for SO_FILE in "${ARROW_LIB}/libarrow.so" "${ARROW_LIB}/libarrow_python.so"; do
  if [ ! -f "$SO_FILE" ]; then
    echo "$SO_FILE does not exist! Trying to create a symlink"
    ln -sf "$(ls "$SO_FILE".*)" "$SO_FILE" || exit 1
  fi
  done

  echo "SOURCE_DIR: ${SOURCE_DIR}"
  BUILD_PATH=$(pwd)/build
  mkdir -p ${BUILD_PATH}
  pushd ${BUILD_PATH} || exit 1
  cmake -DPYCYLON_BUILD=${PYTHON_BUILD} -DCMAKE_BUILD_TYPE=${BUILD_MODE} \
      -DCYLON_WITH_TEST=${RUN_CPP_TESTS} -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
      -DARROW_BUILD_TYPE="SYSTEM" -DARROW_LIB_DIR=${ARROW_LIB} -DARROW_INCLUDE_DIR=${ARROW_INC} \
      -DGCYLON_BUILD=${GCYLON_BUILD} \
      ${CMAKE_FLAGS} \
      ${SOURCE_DIR} \
      || exit 1
  make -j ${MAKE_JOBS} || exit 1
  printf "Cylon CPP Built Successfully!"
  make install || exit 1
  printf "Cylon CPP Installed Successfully!"
  popd || exit 1
  print_line
}

build_gcylon(){
  print_line
  echo "Building GCylon in ${BUILD_MODE} mode"
  print_line

  # set install path to conda directory if not already set
  INSTALL_PATH=${INSTALL_PATH:=${PREFIX:=${CONDA_PREFIX}}}

  echo "SOURCE_DIR: ${SOURCE_DIR}"
  BUILD_PATH=$(pwd)/build
  mkdir -p ${BUILD_PATH}
  pushd ${BUILD_PATH} || exit 1
  cmake -DCMAKE_BUILD_TYPE=${BUILD_MODE} -DCYLON_WITH_TEST=${RUN_CPP_TESTS} -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DGCYLON_BUILD=${GCYLON_BUILD} ${CMAKE_FLAGS} ${SOURCE_DIR} \
      || exit 1
  make -j ${MAKE_JOBS} || exit 1
  printf "GCylon CPP Built Successfully!"
  make install || exit 1
  printf "GCylon CPP Installed Successfully!"
  popd || exit 1
  print_line
}

build_pyarrow(){
  print_line
  echo "Building PyArrow"
  pushd ${BUILD_PATH} || exit 1
  export ARROW_HOME=${BUILD_PATH}/arrow/install
  popd || exit
  source "${PYTHON_ENV_PATH}"/bin/activate || exit 1
  read_python_requirements
  check_python_pre_requisites
  pushd ${BUILD_PATH}/arrow/arrow/python || exit 1
  PYARROW_CMAKE_OPTIONS="-DCMAKE_MODULE_PATH=${ARROW_HOME}/lib/cmake/arrow" PYARROW_WITH_PARQUET=1 python3 setup.py install || exit 1
  popd || exit 1
  print_line
}

build_python_pyarrow() {
  print_line
  echo "Building Python"
  source "${PYTHON_ENV_PATH}"/bin/activate || exit 1
  read_python_requirements
  pip install pyarrow==5.0.0 || exit 1

  ARROW_LIB=$(python3 -c 'import pyarrow as pa; import os; print(os.path.dirname(pa.__file__))') || exit 1
  LD_LIBRARY_PATH="${ARROW_LIB}:${BUILD_PATH}/lib:${LD_LIBRARY_PATH}" || exit 1
  export_library_path ${LD_LIBRARY_PATH}

  check_python_pre_requisites
  pushd python/pycylon || exit 1
  pip3 uninstall -y pycylon
  make clean
  CYLON_PREFIX=${BUILD_PATH}  python3 setup.py install || exit 1
  popd || exit 1
  print_line
}

build_python_conda() {
  print_line
  echo "Building Python"
  ARROW_LIB=${CONDA_PREFIX}/lib
  LD_LIBRARY_PATH="${ARROW_LIB}:${BUILD_PATH}/lib:${LD_LIBRARY_PATH}" || exit 1
  export_library_path ${LD_LIBRARY_PATH}

  pushd python/pycylon || exit 1
  make clean
  CYLON_PREFIX=${BUILD_PATH} ARROW_PREFIX=${BUILD_PREFIX:=${CONDA_PREFIX}} python3 setup.py install || exit 1
  popd || exit 1
  print_line
}

build_pygcylon() {
  print_line
  echo "Building PyGcylon ............. "

  pushd python/pygcylon || exit 1
  make clean
  python3 setup.py install || exit 1
  echo "Built and installed pygcylon ........................"
  popd || exit 1
  print_line
}

build_python() {
  print_line
  echo "Building Python"
  LD_LIBRARY_PATH=${BUILD_PATH}/arrow/install/lib:${BUILD_PATH}/lib:$LD_LIBRARY_PATH || exit 1
  export_library_path ${LD_LIBRARY_PATH}
  # shellcheck disable=SC1090
  source "${PYTHON_ENV_PATH}"/bin/activate || exit 1
  read_python_requirements
  check_python_pre_requisites
  pushd python/pycylon || exit 1
  pip3 uninstall -y pycylon
  make clean
  CYLON_PREFIX=${BUILD_PATH} ARROW_PREFIX=${BUILD_PATH}/arrow/install python3 setup.py install || exit 1
  popd || exit 1
  print_line
}


release_python() {
  print_line
  echo "Building Python"
  LD_LIBRARY_PATH=${BUILD_PATH}/arrow/install/lib:${BUILD_PATH}/lib:$LD_LIBRARY_PATH
  export_library_path ${LD_LIBRARY_PATH}
  source "${PYTHON_ENV_PATH}"/bin/activate
  read_python_requirements
  check_python_pre_requisites
  pushd python/pycylon || exit 1
  pip3 uninstall -y pycylon
  make clean
  # https://www.scivision.dev/easy-upload-to-pypi/ [solution to linux wheel issue]
  #ARROW_HOME=${BUILD_PATH} python3 setup.py sdist bdist_wheel
  ARROW_HOME=${BUILD_PATH} python3 setup.py build_ext --inplace --library-dir=${BUILD_PATH} || exit 1
  popd || exit 1
  print_line
}

export_info(){
  print_line
  echo "Add the following to your LD_LIBRARY_PATH";
  if [ "${PYTHON_WITH_PYARROW_BUILD}" = "ON" ]; then
    echo "export LD_LIBRARY_PATH=${BUILD_PATH}/lib:\$LD_LIBRARY_PATH";
  else
    echo "export LD_LIBRARY_PATH=${BUILD_PATH}/arrow/install/lib:${BUILD_PATH}/lib:\$LD_LIBRARY_PATH";
  fi
  print_line
}

check_pyarrow_installation(){
  LD_LIBRARY_PATH=${BUILD_PATH}/arrow/install/lib:${BUILD_PATH}/lib:$LD_LIBRARY_PATH
  export_library_path ${LD_LIBRARY_PATH}
  response=$(python3 -c \
    "import pyarrow; print('PyArrow Installation');\
    print('Version {}'.format(pyarrow.__version__));\
    print('Library Installation Path {}'.format(pyarrow.get_library_dirs()))")
  echo "${response}"
}

check_pycylon_installation(){
  response=$(python3 python/pycylon/test/test_pycylon.py)
  echo "${response}"
}

python_test(){
  LD_LIBRARY_PATH="${BUILD_PATH}/lib:${BUILD_PATH}/arrow/install/lib" || exit 1
  export_library_path ${LD_LIBRARY_PATH}
  ARROW_LIB=$(python3 -c 'import pyarrow as pa; import os; print(os.path.dirname(pa.__file__))') || exit 1
  LD_LIBRARY_PATH="${ARROW_LIB}:${LD_LIBRARY_PATH}" || exit 1
  export_library_path ${LD_LIBRARY_PATH}
  python3 -m pytest python/pycylon/test/test_all.py || exit 1
}

pygcylon_test(){
  python3 -m pytest python/pygcylon/test/test_all.py || exit 1
}

build_java(){
  echo "Building Java"
  cd java
  mvn clean install -Dcylon.core.libs=$BUILD_PATH/lib -Dcylon.arrow.dir=$BUILD_PATH/arrow/ || exit 1
  echo "Cylon Java built Successufully!"
  cd ../
}

####################################################################################################

if [ "${BUILD_MODE_DEBUG}" = "ON" ]; then
   	BUILD_MODE=Debug	
fi

if [ "${BUILD_MODE_RELEASE}" = "ON" ]; then
   	BUILD_MODE=Release	
fi

if [ "${CPP_BUILD}" = "ON" ]; then
   	build_cpp
fi

if [ "${PYTHON_BUILD}" = "ON" ]; then
  if [ -z "$PYTHON_ENV_PATH" ]; then
    echo "To build python, -pyenv|--python_env_path should be set to a python environment"
    exit 1
  fi
fi

if [ "${PYTHON_BUILD}" = "ON" ]; then
	export_info
	build_pyarrow
	check_pyarrow_installation
	build_python
	check_pycylon_installation
fi

if [ "${PYTHON_WITH_PYARROW_BUILD}" = "ON" ]; then
	export_info
	build_cpp_with_custom_arrow
	build_python_pyarrow
fi


if [ "${CYTHON_BUILD}" = "ON" ]; then
	export_info
	if [ "${PYTHON_WITH_PYARROW_BUILD}" = "ON" ]; then
	  build_python_pyarrow
	else
	  build_python
	fi
	check_pycylon_installation
fi

if [ "${JAVA_BUILD}" = "ON" ]; then
	build_java
fi

if [ "${PYTHON_RELEASE}" = "ON" ]; then	
	export_info
	check_pyarrow_installation
	release_python
fi

if [ "${CONDA_CPP_BUILD}" = "ON" ]; then
	echo "Running conda build"
	build_cpp_conda
fi

if [ "${CONDA_PYTHON_BUILD}" = "ON" ]; then
	echo "Running conda build"
	build_python_conda
fi

if [ "${GCYLON_BUILD}" = "ON" ]; then
	echo "Running conda build"
	build_gcylon
fi

if [ "${PYGCYLON_BUILD}" = "ON" ]; then
	echo "Running pygcylon conda build"
	build_pygcylon
fi

if [ "${RUN_CPP_TESTS}" = "ON" ]; then
	echo "Running CPP tests"
	CTEST_OUTPUT_ON_FAILURE=1 make -C "$BUILD_PATH" test || exit 1
fi

if [ "${RUN_PYTHON_TESTS}" = "ON" ]; then
	echo "Running Python tests"
	python_test
fi

if [ "${RUN_PYGCYLON_TESTS}" = "ON" ]; then
	echo "Running PyGcylon tests"
	pygcylon_test
fi



