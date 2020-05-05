export ARROW_HOME=$(pwd)/cpp/build/arrow/install
source cpp/build/ENV/bin/activate
cd cpp/build/arrow/arrow/python
PYARROW_CMAKE_OPTIONS="-DCMAKE_MODULE_PATH=/home/vibhatha/github/forks/twisterx/cpp/build/arrow/install/lib/cmake/arrow" python3 setup.py install
