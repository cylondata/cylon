---
id: python_table_docs
title: PyCylon Table Docs
---

## Imports

```python
>>> from pycylon import Table
>>> from pycylon import CylonContext
```

## Initialize Context

```python
>>>  ctx: CylonContext = CylonContext(config=None, distributed=False)
```

## Initialize Table

### Using a List

```python
>>> Table.from_list(ctx, ['col-1', 'col-2', 'col-3'], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12
```

### Using a Dictionary

```python
>>> Table.from_pydict(ctx, {'col-1': [1, 2, 3, 4], 'col-2': [5, 6, 7, 8], 'col-3': [9, 10, 11, 12]})
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12
```

### Using a PyArrow Table

```python
>>> atb
    pyarrow.Table
    col-1: int64
    col-2: int64
    col-3: int64

>>> Table.from_arrow(ctx, atb)
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12
```

### Using Numpy

```python
>>> Table.from_numpy(ctx, ['c1', 'c2', 'c3'], [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]), np.array([9, 10, 11, 12])])
       c1  c2  c3
    0   1   5   9
    1   2   6  10
    2   3   7  11
    3   4   8  12
```

### Using Pandas

```python
>>> df
        col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> Table.from_pandas(ctx, df)
        col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12
```

## Convert Table

```python
>>> tb
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

```

### To a PyArrow Table

```python
>>> tb.to_arrow()
pyarrow.Table
col-1: int64
col-2: int64
col-3: int64
```

### To Pandas

```python
>>> tb.to_pandas()
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12
```

### To Numpy

Add `order` as `F` or `C` to get `F_CONTIGUOUS` or `C_CONTIGUOUS` Numpy array.
The default does a zero copy. But for bool values make sure to add `zero_copy_only`
to `False`.

```python
>>> tb.to_numpy(order='F')
[[ 1  5  9]
 [ 2  6 10]
 [ 3  7 11]
 [ 4  8 12]]
```

### To Dictionary

```python
>>> tb.to_pydict()
    {'col-1': [1, 2, 3, 4], 'col-2': [5, 6, 7, 8], 'col-3': [9, 10, 11, 12]}
```

## I/O Operations

### Read from CSV

```python
>>> from pycylon.io import CSVReadOptions
>>> from pycylon.io import read_csv
>>> csv_read_options = CSVReadOptions().with_delimiter('::').use_threads(True).block_size(1 << 30)
>>> read_csv(ctx, '/tmp/data.csv', csv_read_options)
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12
```

### Write to CSV

```python
>>> from pycylon.io import CSVWriteOptions

>>> csv_write_options = CSVWriteOptions().with_delimiter(',')
>>> tb.to_csv('/tmp/data.csv', csv_write_options)
```

## Properties

```python
>>> tb
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

```

### Column Names

```python
>>> tb.column_names
    ['col-1', 'col-2', 'col-3']
```

### Column Count

```python
>>> tb.column_count
    3
```

### Shape

```python
>>> tb.shape
    (4, 3)
```

### Row Count

```python
>>> tb.row_count
```

### Context

```python
>>> tb.context
    <pycylon.ctx.context.CylonContext object at 0x7fb4f4d301e0>
```

## Relational Algebra Operators

```python
>>> tb = Table.from_pydict(ctx, {'keyA': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
      keyA   A
    0   K0  A0
    1   K1  A1
    2   K2  A2
    3   K3  A3
    4   K4  A4
    5   K5  A5
>>> other = Table.from_pydict(ctx, {'keyB': ['K0', 'K1', 'K2'],
                                       'B': ['B0', 'B1', 'B2']})
      keyB   B
    0   K0  B0
    1   K1  B1
    2   K2  B2
```

### Join

1. Join type can be : `join_type` => 'left', 'right', 'inner', 'outer'
2. Join algorithm can be : `algorithm` => 'hash', 'sort'
3. Join on, 'on' when common column is there, otherwise 'left_on' and 'right_on'

Note: The print methods are work in progress to provide similar output as Pandas

In sequential setting use `join` and in distributed setting use `distributed_join` upon the
use-case.

```python
>>> tb.join(table=other, join_type='left', algorithm='sort', left_on=['keyA'], right_on=[
    'keyB'])
      keyA   A keyB   B
    0   K0  A0   K0  B0
    1   K1  A1   K1  B1
    2   K2  A2   K2  B2
    3   K3  A3
    4   K4  A4
    5   K5  A5
```

### Subtract (Difference)

For distributed operations use `distributed_subtract` instead of `subtract`.

```python
>>> tb = Table.from_pydict(ctx, {'keyA': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
      keyA   A
    0   K0  A0
    1   K1  A1
    2   K2  A2
    3   K3  A3
    4   K4  A4

>>> other = other: Table = Table.from_pydict(ctx, {'keyB': ['K0', 'K1', 'K2'],
                                       'B': ['A0', 'A1', 'A2']})
      keyB   B
    0   K0  A0
    1   K1  A1
    2   K2  A2

>>> tb.subtract(other)
      keyA   A
    0   K5  A5
    1   K4  A4
    2   K3  A3
```

### Intersect

For distributed operations use `distributed_intersect` instead of `intersect`.

```python
>>> tb = Table.from_pydict(ctx, {'keyA': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
      keyA   A
    0   K0  A0
    1   K1  A1
    2   K2  A2
    3   K3  A3
    4   K4  A4

>>> other = other: Table = Table.from_pydict(ctx, {'keyB': ['K0', 'K1', 'K2'],
                                       'B': ['A0', 'A1', 'A2']})
      keyB   B
    0   K0  A0
    1   K1  A1
    2   K2  A2

>>> tb.intersect(other)
      keyA   A
    0   K2  A2
    1   K1  A1
    2   K0  A0
```

### Project

For distributed operations and sequential operations `project` can be used.

```python
>>> tb = Table.from_pydict(ctx, {'keyA': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
      keyA   A
    0   K0  A0
    1   K1  A1
    2   K2  A2
    3   K3  A3
    4   K4  A4

>>> tb.project(['A'])
        A
    0  A0
    1  A1
    2  A2
    3  A3
    4  A4
    5  A5
```



