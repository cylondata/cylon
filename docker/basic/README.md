# Cylon Docker Image 

## Instructions 

- To build the Docker image, 
```bash
docker build -t cylon .
```

- To run a container
```bash
docker run -it cylon
```

Cylon will be installed in `/cylon/` directory. A container would have the following environment 
variables and commands. 
- `$CYLON_HOME` points to `/cylon`
- `$CYLON_BUILD` pointst to c++ build directory
- `$CYLON_ENV` points to python virtual environment
- `build_cylon` an alias to rebuild the project that expands to, 
```bash
cd $CYLON_HOME && ./build.sh -pyenv $CYLON_ENV -bpath $CYLON_BUILD --cpp --python --release
```
- `build_cylon_test` an alias to rebuild the project that expands to, 
```bash
cd $CYLON_HOME && ./build.sh -pyenv $CYLON_ENV -bpath $CYLON_BUILD --cpp --test --python --pytest --release 
```

## Running Cylon Examples

Start a container.

```bash
docker run -it cylondata/cylon
```

Execute a python example locally

```bash
python3 /cylon/python/examples/dataframe/data_loading.py /cylon/data/input/csv1_0.csv
```
