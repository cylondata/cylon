# Cylon Docker Image 

## Instructions 

- To build the Docker image, 
```bash
docker build -t cylon-ucc-redis .
```

- To run a container
```bash
docker run -it --name cylon1 -p 22:22 -d cylon-ucc-redis
```

- To attach to a container
```bash
 docker attach cylon1
```



Cylon will be installed in `/cylon/` directory. A container would have the following environment 
variables and commands. 
- `$CYLON_HOME` points to `/cylon`
- `$CYLON_BUILD` points to c++ build directory
- `$CYLON_ENV` points to python virtual environment

UCX will be installed in `/opt/conda/envs/cylon_dev/` directory as part of the environment creation.

UCC with be installed into  `/ucc/` directory. A container would have the following environment 
variables and commands. 
- `$UCC_HOME` points to `/ucc`

Hiredis will be be installed in '/hiredis' directory.

Redis Plus Plus will be installed in `/redis-plus-plus` directory.