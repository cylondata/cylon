---
id: 0.4.0
title: Cylon Release 0.4.0
sidebar_label: Cylon Release 0.4.0
---

Cylon 0.4.0 is a major release with the following features.

## Major Features

### Python
- DataFrame API similar to Pandas supporting around 40 operators commonly used in Pandas.  
- Conda build and conda based binaries for Linux for installing.
- Python binding to all the operators added on the C++ level.
- Providing compute functions with both Arrow and Numpy for filtering, math operations and comparison operators. 
- Added operator benchmarks.
- Added new options for CSV reading supporting all the options in PyArrow for reading CSV.

### C++ 
- Added distributed multi-column operations on tables for join, union, intersection, set difference and sort.
- Added improved hash operations using Bytell Hash Maps. Improved performance by 2 times for union, intersection, set difference and unique.
- Added new aggregate operations for GroupBy operation (Mean, Variance, Std Dev, Quantile, NUnique, Median).
- Implemented GroupBy aggregators using CRTP (Curiously recurring template pattern).
- Improved indexing at the core by Added more types, improved performance of indexed lookups.
- Added unique distributed operator.
- Added temporal data types like DateTime, Date32 (seconds resolution), Date64 (milliseconds resolution) and TImestamp (with time zone information).
- Other performance improvements and bug fixes. 

### Build
- Compiling using external Apache Arrow installation (local/ pip).

### Applications and Benchmarks
- Implementing a subset of TPC-XBB queries (Queries 6, 7, 9, 14, 22, 23) and the rest is ongoing.
- Applications with connections to deep learning.

You can download source code from [Github](https://github.com/cylondata/cylon/releases)

## License

Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
