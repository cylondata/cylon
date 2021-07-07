---
id: quick_start
title: Quick Start
---

Cylon has a few heavy dependencies such as OpenMPI and Apache Arrow that needs to be compiled and configured 
before start writing Cylon applications. While Cylon build takes care of these things for you, optionally 
following approaches can be used to quick start with the development process.

# Get started with Docker

Cylon docker images contains prebuilt cylon binaries and environment configured to start development right away.

Start by creating a volume to hold you source code. 

```bash
docker volume create cylon-vol
```

Then start a Cylon container as follows with your new volume mounted at /code.

```bash
docker run -it -v cylon-vol:/code cylondata/cylon
```

Optionally, you could skip creating a volume and mount a local folder to the /code directory.

You could either develop directly inside the container using a supported IDE to connect to the container(Recommended)
or you could develop on the host machine and test inside the container.


# Get started with Conda



