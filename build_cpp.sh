BUILD_HOME=cpp/build
mkdir ${BUILD_HOME}
pushd ${BUILD_HOME}
python3 -m venv ENV
source ENV/bin/activate
pip3 install numpy cython pandas
export ARROW_HOME=$(pwd)/${BUILD_HOME}/arrow/install
echo "ARROW HOME SET :"${ARROW_HOME}
cmake -DCMAKE_BUILD_TYPE=Release ../
make -j 4
popd

