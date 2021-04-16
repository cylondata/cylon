---
id: 0.4.0
title: Cylon Release 0.4.0
sidebar_label: Cylon Release 0.4.0
---

Cylon 0.4.0 is a major release with the following features.

## Major Features

### Build
- Conda build and conda based binaries
- Compiling using external Apache Arrow installation (local/ pip) 

### C++
- Adding distributed multi-column operations on tables for join, set operations and sort 
- Adding improved hash operations using third part Bytell Hash Maps
- Adding new aggregate operations for GroupBy operation (Mean, Variance, Std Dev, Quantile, NUnique, Median)
- Implementing hash aggregate kernels using CRTP pattern
- Adding improved C++ indexing
- Adding Unique operator kernels
- Adding temporal data types 
- Performance improvements and bug fixes 

### Python

- Supporting all the operators added on the C++ level
- Adding DataFrame API similar to Pandas 
- Providing compute functions with both Arrow and Numpy for filtering, math operations and comparison operators. 
- Adding new operators for dataframe (astype, unique)
- Adding operator benchmarks
- Adding new options for CSV reading

### Other

- Implementing a subset of TPC-XBB queries (Queries 6, 7, 9, 14, 22, 23)

You can download source code from [Github](https://github.com/cylondata/cylon/releases)

## License

Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
