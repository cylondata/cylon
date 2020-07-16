SOURCE_DIR=$(pwd)/cpp
CPP_BUILD="OFF"
PYTHON_BUILD="OFF"
PYARROW_BUILD="OFF"
BUILD_ALL="OFF"
BUILD_MODE=Debug
BUILD_MODE_DEBUG="OFF"
BUILD_MODE_RELEASE="OFF"
PYTHON_RELEASE="OFF"
INSTALL_PATH=

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
    PYTHON_BUILD="ON"
    shift # past argument
    ;;
    --pyarrow)
    PYARROW_BUILD="ON"
    shift # past argument
    ;;
    --debug)
    BUILD_MODE_DEBUG="ON"
    shift # past argument
    ;;
    --release)
    BUILD_MODE_RELEASE="ON"
    shift # past argument
    ;;
    --py-release)
    PYTHON_RELEASE="ON"
    CPP_BUILD="OFF"
    PYARROW_BUILD="OFF"
    shift # past argument
    ;;
    --all)
    BUILD_ALL="ON"
    CPP_BUILD="ON"
    PYTHON_BUILD="ON"
    PYARROW_BUILD="ON"
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo "PYTHON ENV PATH    = ${PYTHON_ENV_PATH}"
echo "BUILD PATH         = ${BUILD_PATH}"
echo "FLAG CPP BUILD     = ${CPP_BUILD}"
echo "FLAG PYTHON BUILD  = ${PYTHON_BUILD}"
echo "FLAG BUILD ALL     = ${BUILD_ALL}"
echo "FLAG BUILD DEBUG   = ${BUILD_MODE_DEBUG}"
echo "FLAG BUILD RELEASE = ${BUILD_MODE_RELEASE}"
#echo "Number files in SEARCH PATH with EXTENSION:" $(ls -1 "${SEARCHPATH}"/*."${EXTENSION}" | wc -l)

if [[ -n $1 ]]; then
    echo "Last line of file specified as non-opt/last argument:"
fi

################################## Util Functions ##################################################
quit() {
exit
}

print_line() {
echo "=================================================================";
}

read_python_requirements(){
input="requirements.txt"
while IFS= read -r line
do
  pip3 install "$line"
done < "$input"
}

INSTALL_CMD=
if [ -z "$INSTALL_PATH" ]
then
  echo "\-ipath|--install_path is NOT set default to cmake"
else
  INSTALL_CMD="-DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}"
  echo "Install location set to: ${INSTALL_PATH}"
fi

build_cpp(){
print_line
echo "Building CPP in ${BUILD_MODE} mode" 
print_line
mkdir ${BUILD_PATH}
pushd ${BUILD_PATH}
export ARROW_HOME=${BUILD_PATH}/arrow/install
cmake -DPYARROW_BUILD=${PYARROW_BUILD} -DPYCYLON_BUILD=${PYTHON_BUILD} -DPYTHON_EXEC_PATH=${PYTHON_ENV_PATH} -DCMAKE_BUILD_TYPE=${BUILD_MODE} $INSTALL_CMD ${SOURCE_DIR}
make -j 4
printf "\n\n ### ARROW HOME SET :%s \n\n" "${ARROW_HOME}"
printf "\n\n ### Cylon CPP Built Successufully !!! \n\n"
popd
print_line
}

build_pyarrow(){
print_line
echo "Building PyArrow"
pushd ${BUILD_PATH}
export ARROW_HOME=${BUILD_PATH}/arrow/install
popd
source ${PYTHON_ENV_PATH}/bin/activate
read_python_requirements
pushd ${BUILD_PATH}/arrow/arrow/python
PYARROW_CMAKE_OPTIONS="-DCMAKE_MODULE_PATH=${ARROW_HOME}/lib/cmake/arrow" python3 setup.py install
popd
print_line
}


build_python() {
print_line
echo "Make sure CPP build is already done!"
echo "Building Python"

export LD_LIBRARY_PATH=${BUILD_PATH}/arrow/install/lib:${BUILD_PATH}/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH="$LD_LIBRARY_PATH
source ${PYTHON_ENV_PATH}/bin/activate
read_python_requirements
pushd python
pip3 uninstall -y pycylon
make clean
ARROW_HOME=${BUILD_PATH} python3 setup.py install
popd
print_line
}

release_python() {
print_line
echo "Make sure CPP build is already done!"
echo "Building Python"

export LD_LIBRARY_PATH=${BUILD_PATH}/arrow/install/lib:${BUILD_PATH}/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH="$LD_LIBRARY_PATH
source ${PYTHON_ENV_PATH}/bin/activate
read_python_requirements
pushd python
pip3 uninstall -y pycylon
make clean
# https://www.scivision.dev/easy-upload-to-pypi/ [solution to linux wheel issue]
#ARROW_HOME=${BUILD_PATH} python3 setup.py sdist bdist_wheel
ARROW_HOME=${BUILD_PATH} python3 setup.py build_ext --inplace --library-dir=${BUILD_PATH}
popd
print_line
}

export_info(){
print_line
echo "Add the following to your LD_LIBRARY_PATH";
echo "export LD_LIBRARY_PATH=${BUILD_PATH}/arrow/install/lib:${BUILD_PATH}/lib:"\$"LD_LIBRARY_PATH";
print_line
}

check_pyarrow_installation(){
response=$(python3 -c \
	"import pyarrow; print('PyArrow Installation');\
	print('Version {}'.format(pyarrow.__version__));\
	print('Library Installation Path {}'.format(pyarrow.get_library_dirs()))")
echo "${response}"
}

check_pycylon_installation(){
response=$(python3 python/test/test_pycylon.py)
echo "${response}"
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


if [ "${PYARROW_BUILD}" = "ON" ]; then
	build_cpp
	build_pyarrow
	check_pyarrow_installation
	export_info

fi	


if [ "${PYTHON_BUILD}" = "ON" ]; then	
	export_info
	check_pyarrow_installation
	build_python
	check_pycylon_installation
fi

if [ "${PYTHON_RELEASE}" = "ON" ]; then	
	export_info
	check_pyarrow_installation
	release_python
fi



