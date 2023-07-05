FROM --platform=linux/amd64 ubuntu:22.04
MAINTAINER cylondata@googlegroups.com

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

ENV CYLON_HOME=/cylon
ENV CYLON_PREFIX=/cylon/install
ENV UCC_HOME=/ucc

WORKDIR $CYLON_HOME

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    #/opt/conda/bin/conda clean -a && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate cylon_dev" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CYLON_PREFIX" >> ~/.bashrc




RUN git clone https://github.com/mstaylor/cylon.git $CYLON_HOME
RUN cd $CYLON_HOME && git checkout -b aws-docker-support origin/aws-docker-support

RUN conda install -n base conda-libmamba-solver && conda config --set solver libmamba && \
    conda install -c anaconda boto3 && pip install cloudmesh-openstack

RUN conda env create -f $CYLON_HOME/conda/environments/cylon.yml

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update &&  apt-get install autoconf -y &&  apt-get install libtool -y && apt-get install cmake -y && \
    apt-get install libnuma-dev -y

#UCC install

RUN git clone --single-branch -b master https://github.com/openucx/ucc.git $UCC_HOME
RUN cd $UCC_HOME && \
   ./autogen.sh && \
   ./configure --prefix=$UCC_HOME/install --with-ucx=/opt/conda/envs/cylon_dev && \
   make install

#REDIS install
RUN git clone https://github.com/redis/hiredis.git /hiredis && \
    cd /hiredis && \
    make && \
    make install && \
    git clone https://github.com/sewenew/redis-plus-plus.git /redis-plus-plus && \
    cd /redis-plus-plus && \
    mkdir build && \
    cd build && \
    cmake -DREDIS_PLUS_PLUS_CXX_STANDARD=11 .. && \
    make && make install


RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    cd $CYLON_HOME && \
    python build.py -cmake-flags="-DCYLON_UCX=1 -DCYLON_UCC=1 -DUCC_INSTALL_PREFIX=$UCC_HOME/install -DCYLON_USE_REDIS=1 " -ipath="$CYLON_PREFIX" --cpp --python --test --pytest


ENTRYPOINT ["/bin/bash"]
