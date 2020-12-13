---
id: 0.3.0
title: Cylon Release 0.3.0
sidebar_label: Cylon Release 0.3.0
---

Cylon 0.3.0 adds the following features. Please note that this release may not be backward
 compatible with previous releases.

## Major Features

### C++

- Adding order-by and distributed table sort operations
- Multiple partitioning schemes (modulo, hash and range)
- C++ API refactoring
- Performance improvements in the existing C++ API

### Python (Pycylon)

- Exposing table operators similar to Pandas (28 new operators).
  - Comparison operators
  - Logical Operators
  - Math operators
  - Null/NA value filtering and filling
  - Filtering and updating (including inplace ops)
  - Schema refactoring
  - Experimental indexing abstract
- Distributed Data sorting Python bindings
- Adding new examples for updated operations. (https://github.com/cylondata/cylon/tree/master/python/examples)


You can download source code from [Github](https://github.com/cylondata/cylon/releases)

## Examples

- [C++ examples](https://github.com/cylondata/cylon/tree/0.3.0/cpp/src/examples)
- [Python examples](https://github.com/cylondata/cylon/tree/0.3.0/python/examples)
- [Java examples](https://github.com/cylondata/cylon/tree/0.3.0/java/src/main/java/org/cylondata/cylon/examples)

## License

Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0