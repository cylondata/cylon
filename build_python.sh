export LD_LIBRARY_PATH=$(pwd)/cpp/build/arrow/install/lib:$(pwd)/cpp/build/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH="$LD_LIBRARY_PATH
source cpp/build/ENV/bin/activate
cd python
sudo make clean
pip3 uninstall -y pytwisterx
python3 setup.py install
