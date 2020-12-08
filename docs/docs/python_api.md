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

>>> tb: Table = Table.from_arrow(ctx, atb)
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

