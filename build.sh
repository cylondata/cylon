SOURCE_DIR=$(pwd)/cpp
FLAG_CPP_BUILD=0
FLAG_PYTHON_BUILD=0
FLAG_PYARROW_BUILD=0
BUILD_ALL=0
BUILD_MODE=Debug
BUILD_MODE_DEBUG=0
BUILD_MODE_RELEASE=0

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
    --cpp)
    FLAG_CPP_BUILD=1
    shift # past argument
    ;;
    --python)
    FLAG_PYTHON_BUILD=1
    shift # past argument
    ;;
    --pyarrow)
    FLAG_PYARROW_BUILD=1
    shift # past argument
    ;;
    --debug)
    BUILD_MODE_DEBUG=1
    shift # past argument
    ;;
    --release)
    BUILD_MODE_RELEASE=1
    shift # past argument
    ;;
    --all)
    BUILD_ALL=1
    FLAG_CPP_BUILD=1
    FLAG_PYTHON_BUILD=1
    FLAG_PYARROW_BUILD=1
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo "FILE EXTENSION     = ${PYTHON_ENV_PATH}"
echo "SEARCH PATH        = ${BUILD_PATH}"
echo "FLAG CPP BUILD     = ${FLAG_CPP_BUILD}"
echo "FLAG PYTHON BUILD  = ${FLAG_PYTHON_BUILD}"
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

build_cpp(){
print_line
echo "Building CPP in ${BUILD_MODE} mode" 
print_line
mkdir ${BUILD_PATH}
pushd ${BUILD_PATH}
export ARROW_HOME=${BUILD_PATH}/arrow/install
cmake -DPYTHON_EXEC_PATH=${PYTHON_ENV_PATH} -DCMAKE_BUILD_TYPE=${BUILD_MODE} ${SOURCE_DIR}
make -j 4
printf "\n\n ### ARROW HOME SET :%s \n\n" "${ARROW_HOME}"
printf "\n\n ### TwisterX CPP Built Successufully !!! \n\n"
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
pip3 uninstall -y pytwisterx
make clean
ARROW_HOME=${BUILD_PATH} python3 setup.py install
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

check_pytwisterx_installation(){
response=$(python3 python/test/test_pytwisterx.py)
echo "${response}"
}

####################################################################################################

if [ "${BUILD_MODE_DEBUG}" -eq "1" ]; then
   	BUILD_MODE=Debug	
fi

if [ "${BUILD_MODE_RELEASE}" -eq "1" ]; then
   	BUILD_MODE=Release	
fi


if [ "${FLAG_CPP_BUILD}" -eq "1" ]; then
   	build_cpp
   	if [ "${FLAG_PYARROW_BUILD}" -eq "1" ]; then
   		build_pyarrow
   		export_info
   	fi	
fi

if [ "${FLAG_PYTHON_BUILD}" -eq "1" ]; then	
	export_info
	check_pyarrow_installation
	build_python
	check_pytwisterx_installation
fi




