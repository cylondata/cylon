BUILD_HOME=cpp/build
mkdir ${BUILD_HOME}
pushd ${BUILD_HOME}
python3 -m venv ENV
source ENV/bin/activate
pip3 install numpy cython pandas
export ARROW_HOME=$(pwd)/arrow/install
printf "\n\n ### ARROW HOME SET :%s \n\n" "${ARROW_HOME}"
cmake -DCMAKE_BUILD_TYPE=Release ../
printf "\n\n ### CMAKE DONE!!! \n\n"
make -j 4
popd

