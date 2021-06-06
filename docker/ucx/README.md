# Cylon Docker Image 

## Instructions 

- To build the Docker image, 
```bash
docker build -t cylon .
```

- To run a container
```bash
docker run -it --name cylon1 -p 22:22 -d cylon
```

- To attach to a container
```bash
 docker attach cylon1
```

- Set up ssh for MPI
```bash
sudo adduser cylon1 # create a new user
sudo usermod -aG sudo cylon1 # give root privileges
su - cylon1 # switch to the new user

ssh-keygen -t rsa # generate the ssh key
ssh-copy-id -i /home/cylon1/.ssh/id_rsa.pub 172.17.0.2 #send ssh key to self
ssh-copy-id -i /home/cylon1/.ssh/id_rsa.pub 172.17.0.3 #send ssh key to other node 1
..... # send ssh key to all other nodes
```

Cylon will be installed in `/cylon/` directory. A container would have the following environment 
variables and commands. 
- `$CYLON_HOME` points to `/cylon`
- `$CYLON_BUILD` points to c++ build directory
- `$CYLON_ENV` points to python virtual environment

UCX will be installed in `/ucx/` directory. A container would have the following environment
variables and commands.
- `$UCX_HOME` points to `/ucx`
- `$UCX_SOURCE` points to the source code location (used for building)
- `$UCX_LIB` points to UCX lib directory
- `$UCX_INCLUDE` points to UCX include directory
