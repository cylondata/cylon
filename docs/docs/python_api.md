---
id: python_table_docs
title: PyCylon Table Docs
---

## Imports

```python
>>> from pycylon import Table
>>> from pycylon import CylonContext
```

## Context

Initializing the Cylon Context based on the distributed or non-distributed context
        Args:
            config: an object extended from pycylon.net.CommConfig, pycylon.net.MPIConfig for MPI
            backend
            distributed: bool to set distributed setting True or False
        Returns: None

### Sequential Programming

```python
>>>  ctx: CylonContext = CylonContext(config=None, distributed=False)
```

### Distributed Programmging

```python
>>> from pycylon.net import MPIConfig
>>> mpi_config = MPIConfig()
>>> ctx: CylonContext = CylonContext(config=mpi_config, distributed=True)
```

### Rank

This is the process id (unique per process)
        :return: an int as the rank (0 for non distributed mode)

```python
>>> ctx.get_rank()
    1
```

### World Size

This is the total number of processes joined for the distributed task
        :return: an int as the world size  (1 for non distributed mode)

```python
>>> ctx.get_world_size()
    4
```

### Finalize

Gracefully shuts down the context by closing any distributed processes initialization ,etc
        :return: None

```python
>>> ctx.finalize()
```

### Barrier

Calling barrier to sync workers

```python
>>> ctx.barrier()
```

## Initialize Table

### Using a List

Creating a PyCylon table from a list
        Args:
            context: pycylon.CylonContext
            col_names: Column names as a List[str]
            data_list: data as a List of List, (List per column)

        Returns: PyCylon Table

```python
>>> Table.from_list(ctx, ['col-1', 'col-2', 'col-3'], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12
```

### Using a Dictionary

Creating a PyCylon table from a dictionary
        Args:
            context: pycylon.CylonContext
            dictionary: dict object with key as column names and values as a List

        Returns: PyCylon Table

```python
>>> Table.from_pydict(ctx, {'col-1': [1, 2, 3, 4], 'col-2': [5, 6, 7, 8], 'col-3': [9, 10, 11, 12]})
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12
```

### Using a PyArrow Table

Creating a PyCylon table from PyArrow Table
        Args:
            context: pycylon.CylonContext
            pyarrow_table: PyArrow Table

        Returns: PyCylon Table

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

Creating a PyCylon table from numpy arrays
        Args:
            context: pycylon.CylonContext
            col_names: column names as a List
            ar_list: Numpy ndarrays as a list (one 1D array per column)

        Returns: PyCylon Table

```python
>>> Table.from_numpy(ctx, ['c1', 'c2', 'c3'], [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]), np.array([9, 10, 11, 12])])
       c1  c2  c3
    0   1   5   9
    1   2   6  10
    2   3   7  11
    3   4   8  12
```

### Using Pandas

Creating a PyCylon table from Pandas DataFrame
        Args:
            context: cylon.CylonContext
            df: pd.DataFrame
            preserve_index: keep indexes as same as in original DF
            nthreads: number of threads for the operation
            columns: column names, if updated
            safe: safe operation

        Returns: PyCylon Table

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

Creating PyArrow Table from PyCylon table
         Return: PyArrow Table

```python
>>> tb.to_arrow()
pyarrow.Table
col-1: int64
col-2: int64
col-3: int64
```

### To Pandas

Creating Pandas Dataframe from PyCylon Table
        Returns: pd.DataFrame

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

Creating a dictionary from PyCylon table
        Returns: dict object

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

Creating a csv file with PyCylon table data
        Args:
            path: path to file
            csv_write_options: pycylon.io.CSVWriteOptions

        Returns: None

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

Joins two PyCylon tables
        :param table: PyCylon table on which the join is performed (becomes the left table)
        :param join_type: Join Type as str ["inner", "left", "right", "outer"]
        :param algorithm: Join Algorithm as str ["hash", "sort"]
        :kwargs left_on: Join column of the left table as List[int] or List[str], right_on:
        Join column of the right table as List[int] or List[str], on: Join column in common with
        both tables as a List[int] or List[str].
        Return: Joined PyCylon table

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

## Aggregation Operations

Currently supports, Sum, Min, Max, Count

```python
>>> tb = Table.from_pydict(ctx, {'A': [10, 12, 20, 13, 14, 1, 0],
                                    'B': [13, 14, 10, 19, 114, -1, 5]})
        A    B
    0  10   13
    1  12   14
    2  20   10
    3  13   19
    4  14  114
    5   1   -1
    6   0    5

```

## SUM

```python
>>> tb.sum('A')
        A
    0  70

```

## Min

```python
>>> tb.min('A')
       A
    0  0
```

## Max

```python
>>> tb.max('A')
        A
    0  20
```

## Count

```python
>>> tb.count('A')
       A
    0  7
```

## GroupBy

Group by operations support aggregations.

```python
>>> tb
       AnimalId  Max Speed
    0         1      380.0
    1         1      370.0
    2         2       24.0
    3         2       26.0
    4         3       23.1
    5         4      300.1
    6         4      310.2
    7         3       25.2
```

```python
>>> from pycylon.data.aggregates import AggregationOp
>>> tb.groupby(0, [1], [AggregationOp.SUM])
       AnimalId  Max Speed
    0         4      610.3
    1         3       48.3
    2         2       50.0
    3         1      750.0

>>>
```

## Comparison Operators

### Equal

Equal operator for Table
        Args:
            other: can be a numeric scalar or a Table

        Returns: PyCylon Table

```python
>>> tb
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb['col-1'] == 2
       col-1
    0  False
    1   True
    2  False
    3  False

>>> tb == 2
       col-1  col-2  col-3
    0  False  False  False
    1   True  False  False
    2  False  False  False
    3  False  False  False

```

### Not Equal

Not equal operator for Table
        Args:
            other: can be a numeric scalar or Table

        Returns: PyCylon Table

```python
>>> tb
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb3 = tb['col-1'] != 2
       col-1
    0   True
    1  False
    2   True
    3   True

>>> tb4 = tb != 2
       col-1  col-2  col-3
    0   True   True   True
    1  False   True   True
    2   True   True   True
    3   True   True   True
```

### Lesser Than

Lesser than operator for Table
        Args:
            other: can be a numeric scalar or Table

        Returns: PyCylon Table

```python
>>> tb
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb3 = tb['col-1'] < 2
       col-1
    0   True
    1  False
    2  False
    3  False

>>> tb4 = tb < 2
       col-1  col-2  col-3
    0   True  False  False
    1  False  False  False
    2  False  False  False
    3  False  False  False
```

### Greater Than

Greater than operator for Table
        Args:
            other: can be a numeric scalar or Table

        Returns: PyCylon Table

```python
>>> tb
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb3 = tb['col-1'] > 2
        col-1
    0  False
    1  False
    2   True
    3   True

>>> tb4 = tb > 2
       col-1  col-2  col-3
    0  False   True   True
    1  False   True   True
    2   True   True   True
    3   True   True   True
```

### Lesser Than Equal

Lesser than or equal operator for Table
        Args:
            other: can be a numeric scalar or Table

        Returns: PyCylon Table

```python
>>> tb
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb3 = tb['col-1'] <= 2
        col-1
    0   True
    1   True
    2  False
    3  False

>>> tb4 = tb <= 2
       col-1  col-2  col-3
    0   True  False  False
    1   True  False  False
    2  False  False  False
    3  False  False  False
```

### Greater Than Equal

Greater than or equal operator for Table
        Args:
            other: can be a numeric scalar or Table

        Returns: PyCylon Table

```python
>>> tb
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12


>>> tb3 = tb['col-1'] >= 2
       col-1
    0  False
    1   True
    2   True
    3   True

>>> tb4 = tb >= 2
       col-1  col-2  col-3
    0  False   True   True
    1   True   True   True
    2   True   True   True
    3   True   True   True
```

## Logical Operators

### Or

Or operator for Table
        Args:
            other: PyCylon Table

        Returns: PyCylon Table


```python
>>> tb1
       col-1  col-2
    0  False   True
    1   True   True
    2  False  False
    3   True  False

>>> tb2
        col-1  col-2
    0   True  False
    1   True   True
    2  False  False
    3  False   True

>>> tb_or = tb1 | tb2
       col-1  col-2
    0   True   True
    1   True   True
    2  False  False
    3   True   True
```

### And

And operator for Table
        Args:
            other: PyCylon Table

        Returns: PyCylon Table

```python
>>> tb1
       col-1  col-2
    0  False   True
    1   True   True
    2  False  False
    3   True  False

>>> tb2
        col-1  col-2
    0   True  False
    1   True   True
    2  False  False
    3  False   True

>>> tb_or = tb1 & tb2
       col-1  col-2
    0  False  False
    1   True   True
    2  False  False
    3  False  False
```

### Invert

Only support bool valued Tables

Invert operator for Table

         Returns: PyCylon Table

```python
 >>> tb
        col-1  col-2
    0  False   True
    1   True   True
    2  False  False
    3   True  False

>>> ~tb
       col-1  col-2
    0   True  False
    1  False  False
    2   True   True
    3  False   True
```

## Math Operators

Currently support negation, add, subtract, multiply and division on scalar numeric values.

### Negation

Negation operator for Table

         Returns: PyCylon Table

```python
>>> tb
        col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> -tb
       col-1  col-2  col-3
    0     -1     -5     -9
    1     -2     -6    -10
    2     -3     -7    -11
    3     -4     -8    -12
```

### Add

Add operator for Table
         Args:
             other: scalar numeric

         Returns: PyCylon Table

```python
>>> tb
        col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb + 2
       col-1  col-2  col-3
    0      3      7     11
    1      4      8     12
    2      5      9     13
    3      6     10     14
```

### Subtract

Subtract operator for Table
         Args:
             other: scalar numeric

         Returns: PyCylon Table

```python
>>> tb
        col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb - 2
       col-1  col-2  col-3
    0     -1      3      7
    1      0      4      8
    2      1      5      9
    3      2      6     10
```

### Multiply

Multiply operator for Table
         Args:
             other: scalar numeric

         Returns: PyCylon Table

```python
>>> tb
        col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb * 2
       col-1  col-2  col-3
    0      2     10     18
    1      4     12     20
    2      6     14     22
    3      8     16     24
```

### Division

Element-wise division operator for Table
         Args:
             other: scalar numeric

         Returns: PyCylon Table

```python
>>> tb
        col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb / 2
       col-1  col-2  col-3
    0    0.5    2.5    4.5
    1    1.0    3.0    5.0
    2    1.5    3.5    5.5
    3    2.0    4.0    6.0
```

## Drop

drop a column or list of columns from a Table
        Args:
            column_names: List[str]

        Returns: PyCylon Table

```python
>>> tb
        col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb.drop(['col-1'])
       col-2  col-3
    0      5      9
    1      6     10
    2      7     11
    3      8     12
```

## Fillna

Fill not applicable values with a given value
        Args:
            fill_value: scalar

        Returns: PyCylon Table

```python
>>> tb
       col-1  col-2  col-3
    0    1.0    5.0    9.0
    1    NaN    6.0   10.0
    2    3.0    NaN   11.0
    3    4.0    8.0    NaN

>>> tb.fillna(0)
       col-1  col-2  col-3
    0      1      5      9
    1      0      6     10
    2      3      0     11
    3      4      8      0
```

## Where

Experimental version of Where operation.
        Replace values where condition is False
        Args:
            condition: bool Table
            other: Scalar

        Returns: PyCylon Table

```python
>>> tb
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb.where(tb > 2)
        col-1  col-2  col-3
    0    NaN      5      9
    1    NaN      6     10
    2    3.0      7     11
    3    4.0      8     12

>>> tb.where(tb > 2, 10)
       col-1  col-2  col-3
    0     10      5      9
    1     10      6     10
    2      3      7     11
    3      4      8     12
```

## IsNull

Checks for null elements and returns a bool Table
        Returns: PyCylon Table

```python
>>> tb
       col-1  col-2  col-3
    0    1.0    5.0    9.0
    1    NaN    6.0   10.0
    2    3.0    NaN   11.0
    3    4.0    8.0    NaN

>>> tb.isnull()
        col-1  col-2  col-3
    0  False  False  False
    1   True  False  False
    2  False   True  False
    3  False  False   True
```

## IsNA

Check for not applicable values and returns a bool Table
        Returns: PyCylon Table

```python
>>> tb
       col-1  col-2  col-3
    0    1.0    5.0    9.0
    1    NaN    6.0   10.0
    2    3.0    NaN   11.0
    3    4.0    8.0    NaN

>>> tb.isna()
        col-1  col-2  col-3
    0  False  False  False
    1   True  False  False
    2  False   True  False
    3  False  False   True
```

## Not Null

Check the not null values and returns a bool Table
        Returns: PyCylon Table

```python
>>> tb
       col-1  col-2  col-3
    0    1.0    5.0    9.0
    1    NaN    6.0   10.0
    2    3.0    NaN   11.0
    3    4.0    8.0    NaN

>>> tb.notnull()
       col-1  col-2  col-3
    0   True   True   True
    1  False   True   True
    2   True  False   True
    3   True   True  False
```

## Not NA

Checks for not NA values and returns a bool Table
        Returns: PyCylon Table

```python
>>> tb
        col-1  col-2  col-3
    0    1.0    5.0    9.0
    1    NaN    6.0   10.0
    2    3.0    NaN   11.0
    3    4.0    8.0    NaN

>>> tb.notna()
       col-1  col-2  col-3
    0   True   True   True
    1  False   True   True
    2   True  False   True
    3   True   True  False
```

## Rename

Rename a Table with a column name or column names
        Args:
            column_names: dictionary or full list of new column names

        Returns: PyCylon Table

```python
>>> tb
        col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb.rename({'col-1': 'col_1'})
       col_1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb.rename(['c1', 'c2', 'c3'])
       c1  c2  c3
    0   1   5   9
    1   2   6  10
    2   3   7  11
    3   4   8  12
```

## Add Prefix

Adding a prefix to column names
        Args:
            prefix: str

        Returns: PyCylon Table with prefix updated

```python
>>> tb
        col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb.add_prefix('old_')
       old_c1  old_c2  old_c3
    0       1       5       9
    1       2       6      10
    2       3       7      11
    3       4       8      12
```

## Add Suffix

Adding a prefix to column names
        Args:
            prefix: str

        Returns: PyCylon Table with prefix updated

```python
>>> tb
        col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb.add_suffix('_old')
       c1_old  c2_old  c3_old
    0       1       5       9
    1       2       6      10
    2       3       7      11
    3       4       8      12
```

## Index

Retrieve index if exists or provide a range index as default
        Returns: Index object

```python
>>> tb.index
     <pycylon.index.RangeIndex object at 0x7f58bde8e040>
```

## Set Index

Set Index
        Args:
            key: pycylon.Index Object or an object extended from pycylon.Index

        Returns: None

```python
>>> tb
       col-1  col-2  col-3
    0      1      5      9
    1      2      6     10
    2      3      7     11
    3      4      8     12

>>> tb.set_index(['a', 'b', 'c', 'd'])

>>> tb.index
    <pycylon.index.CategoricalIndex object at 0x7fa72c2b6ca0>

>>> tb.index.index_values
    ['a', 'b', 'c', 'd']
```

## DropNa

Drop not applicable values from a Table
        Args:
            axis: 0 for column and 1 for row and only do dropping on the specified axis
            how: any or all, any refers to drop if any value is NA and drop only if all values
            are NA in the considered axis
            inplace: do the operation on the existing Table itself when set to True, the default
            is False and it produces a new Table with the drop update

        Returns: PyCylon Table

```python
>>> tb
       col-1  col-2  col-3
    0    1.0      5    9.0
    1    NaN      6   10.0
    2    3.0      7   11.0
    3    4.0      8    NaN

>>> tb_na.dropna(how='any')
       col-2
    0      5
    1      6
    2      7
    3      8

>>> tb_na.dropna(how='all')
       col-1  col-2  col-3
    0    1.0      5    9.0
    1    NaN      6   10.0
    2    3.0      7   11.0
    3    4.0      8    NaN

>>> tb_na.dropna(axis=1, how='any')
       col-1  col-2  col-3
    0      1      5      9
    1      3      7     11

>>> tb_na.dropna(axis=1, how='all')
       col-1  col-2  col-3
    0    1.0      5    9.0
    1    NaN      6   10.0
    2    3.0      7   11.0
    3    4.0      8    NaN

>>> tb_na
       col-1  col-2  col-3
    0    1.0      5    9.0
    1    NaN      6   10.0
    2    3.0      7   11.0
    3    4.0      8    NaN

>>> tb_na.dropna(axis=1, how='any', inplace=True)
       col-1  col-2  col-3
    0      1      5      9
    1      3      7     11

>>> tb_na
       col-1  col-2  col-3
    0      1      5      9
    1      3      7     11
```