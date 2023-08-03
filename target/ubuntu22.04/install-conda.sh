#
# install miniconda
# https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
#

mkdir tmp
cd tmp
rm -rf ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# yes to licence
# no to init


eval "$(/home/green/miniconda3/bin/conda shell.bash hook)"


git clone https://github.com/cylondata/cylon.git
cd cylon


rm -rf build

conda env create -f conda/environments/cylon.yml
conda activate cylon_dev
time python build.py  --cpp --test --python --pytest
