FROM ubuntu:20.04
MAINTAINER cylondata@googlegroups.com

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate cylon_dev" >> ~/.bashrc


ENV CYLON_HOME=/cylon
ENV CYLON_PREFIX=/cylon/install

WORKDIR $CYLON_HOME

RUN git clone --single-branch https://github.com/cylondata/cylon.git $CYLON_HOME

RUN conda env create -q -f $CYLON_HOME/conda/environments/cylon.yml

RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate cylon_dev && \
    cd $CYLON_HOME && \
    python build.py -ipath="$CYLON_PREFIX" --cpp --python --test --pytest

ENTRYPOINT ["/bin/bash"]
