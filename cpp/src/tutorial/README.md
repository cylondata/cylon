# Instructions 

## C++ 

- Follow [Cylon docs](https://cylondata.org/docs/) for detailed building instructions, but in
 summary,  
```
./build.sh --cpp --release
```

- Run `simple_join.cpp` example 
```
./build/bin/simple_join
```

- For distributed execution using MPI 
```
mpirun -np <procs> ./build/bin/simple_join
```

## Python 

- Activate the python virtual environment 
```
source <CYLON_HOME>/ENV/bin/activate 
```

- Follow [Cylon docs](https://cylondata.org/docs/) for detailed building instructions, but in
 summary,  
 ```
 ./build.sh --pyenv <CYLON_HOME>/ENV --python --release
 ```

- Export `LD_LIBRARY_PATH`
```
export LD_LIBRARY_PATH=<CYLON_HOME>/build/arrow/install/lib:<CYLON_HOME>/build/lib:$LD_LIBRARY_PATH
```

- Run `simple_join.py` script
```
python ./cpp/src/tutorial/simple_join.py
```
- For distributed execution using MPI 
```
mpirun -np <procs> <<CYLON_HOME>/ENV/bin/python ./cpp/src/tutorial/simple_join.py
```

