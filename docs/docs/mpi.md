---
id: mpi
title: Run with MPI
---

Cylon executables can be run with MPI distributions as follows. 

## C++ 

To run the `cpp/src/examples/join_example.cpp`, first compile cylon. 

Then, run the executable,
```bash
mpirun -np <NUM_WORKERS> <CYLON_HOME>/bin/join_example /path/to/csv1 /path/to/csv2
```

## Python 

To run the `python/test/test_dist_rl.py`, 
```bash
mpirun -np <NUM_WORKERS> <PYTHON_ENV>/bin/python3 <CYLON_HOME>/python/test/test_dist_rl.py
```

## Java 

To run the `java/src/main/java/org/cylondata/cylon/examples/DistributedJoinExample.java`, first create a 
fat-jar with the executables
```bash
mvn clean package
``` 

Then run the fat-jar main class,
```bash
mpirun -np <NUM_WORKERS> java -cp <CYLON_HOME>/target/cylon-0.1.0-jar-with-dependencies.jar  org.cylondata.cylon.examples.DistributedJoinExample /path/to/csv1 /path/to/csv2
```