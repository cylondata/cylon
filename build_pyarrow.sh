export ARROW_HOME=$(pwd)/cpp/build/arrow/install
source cpp/build/ENV/bin/activate
pushd cpp/build/arrow/arrow/python
PYARROW_CMAKE_OPTIONS="-DCMAKE_MODULE_PATH=${ARROW_HOME}/lib/cmake/arrow" python3 setup.py install
popd
