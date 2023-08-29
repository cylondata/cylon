#
# install miniconda
# https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
#


#BASE = $HOME

#mkdir -p $BASE

#cd $BASE
#PWD = `pwd`


#mkdir ~/tmp
#cd ~/tmp
#rm -rf ~/miniconda3
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh
## yes to licence
## no to init


eval "$(/home/green/miniconda3/bin/conda shell.bash hook)"

#cd $PWD

# git clone https://github.com/cylondata/cylon.git
#git clone git@github.com:laszewsk/cylon.git
#cd cylon


rm -rf build

conda env create -f conda/environments/cylon.yml
conda activate cylon_dev
time python build.py  --cpp --test --python --pytest

# AMD Ryzen 9 5950X 16-Core Processor
# real	2m29.929s
# user	16m33.882s
# sys	1m53.101s
# storage on NVME Sabrent Rocket 4.0 Plus (RKT4P1.2)


#
# check a test prg
#
time python  python/pycylon/examples/dataframe/groupby.py

#real	0m0.407s
#user	0m0.680s
#sys	0m1.422s

