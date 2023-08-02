FROM ubuntu:20.04
MAINTAINER cylondata@googlegroups.com

RUN DEBIAN_FRONTEND=noninteractive TZ=America/New_York apt-get update -y  && apt-get upgrade -y \
   && apt-get install -y apt-utils tzdata
RUN dpkg-reconfigure tzdata

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-dev python3-venv python3-pip git build-essential libopenmpi-dev vim cmake libbrotli-dev

ENV CYLON_HOME=/cylon
ENV CYLON_BUILD=$CYLON_HOME/build
ENV CYLON_ENV=$CYLON_HOME/ENV

WORKDIR $CYLON_HOME

RUN git clone https://github.com/cylondata/cylon.git $CYLON_HOME
RUN cd $CYLON_HOME && python3 -m venv --system-site-packages $CYLON_ENV

RUN echo 'alias build_cylon="cd $CYLON_HOME && ./build.sh -pyenv $CYLON_ENV -bpath $CYLON_BUILD --cpp --python --release --cmake-flags -DCYLON_PARQUET=ON"' >> $HOME/.bashrc
RUN echo 'alias build_cylon_test="cd $CYLON_HOME && ./build.sh -pyenv $CYLON_ENV -bpath $CYLON_BUILD --cpp --test --python --pytest --release --cmake-flags -DCYLON_PARQUET=ON"' >> $HOME/.bashrc

RUN cd $CYLON_HOME && /bin/bash build.sh -pyenv $CYLON_ENV -bpath $CYLON_BUILD --cpp --python --release  --cmake-flags -DCYLON_PARQUET=ON

# Activating PyCylon Env
ENV LD_LIBRARY_PATH=$CYLON_BUILD/lib/:$CYLON_BUILD/arrow/install/lib/
ENV PATH="$CYLON_ENV/bin:$PATH"

# Adding Cylon User
RUN useradd -ms /bin/bash cylon

# Creating folder for the applications
RUN mkdir /code && chown cylon /code

USER cylon
WORKDIR /home/cylon

ENTRYPOINT ["/bin/bash"]
