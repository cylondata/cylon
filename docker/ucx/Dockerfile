FROM ubuntu:20.04
MAINTAINER cylondata@googlegroups.com

RUN DEBIAN_FRONTEND=noninteractive TZ=America/New_York apt-get update -y  && apt-get upgrade -y \
   && apt-get install -y apt-utils tzdata
RUN dpkg-reconfigure tzdata

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-dev python3-venv python3-pip git build-essential libopenmpi-dev vim cmake wget
RUN DEBIAN_FRONTEND=noninteractive apt install openssh-server sudo -y

ENV CYLON_HOME=/cylon
ENV CYLON_BUILD=$CYLON_HOME/build
ENV CYLON_ENV=$CYLON_HOME/ENV
ENV UCX_HOME=/ucx
ENV UCX_SOURCE=$UCX_HOME/ucx-1.10.1
ENV UCX_LIB=$UCX_SOURCE/install/lib
ENV UCX_INCLUDE=$UCX_SOURCE/install/include

WORKDIR $CYLON_HOME

RUN wget https://github.com/openucx/ucx/releases/download/v1.10.1/ucx-1.10.1.tar.gz -P $UCX_HOME
RUN cd $UCX_HOME && tar xzf ucx-1.10.1.tar.gz
RUN cd $UCX_SOURCE && ./contrib/configure-release --prefix=$PWD/install
RUN cd $UCX_SOURCE && make -j8 install
RUN git clone https://github.com/cylondata/cylon.git $CYLON_HOME

RUN cd $CYLON_HOME && /bin/bash build.sh -bpath $CYLON_BUILD --cpp --release --cmake-flags "-DCYLON_UCX=ON -DUCX_LIBDIR=$UCX_LIB -DUCX_INCLUDEDIR=$UCX_INCLUDE"

ENTRYPOINT ["/bin/bash"]
