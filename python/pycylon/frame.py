##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##


from __future__ import annotations
from typing import Hashable, List, Dict, Literal, Optional, Sequence, Union
from copy import copy
from collections.abc import Iterable
import pycylon as cn
import numpy as np
import pandas as pd
import pyarrow as pa
from pycylon import Series
from pycylon.index import RangeIndex, CategoricalIndex
from pycylon.io import CSVWriteOptions
from pycylon.io import CSVReadOptions

from pycylon import CylonContext
from pycylon.net.mpi_config import MPIConfig


class DataFrame(object):

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,
                 distributed=False):
        self._index = None
        self._columns = []
        self._is_distributed = distributed
        self._context = None
        self._initialize_context()
        self._table = self._initialize_dataframe(
            data=data, index=index, columns=columns, copy=copy)

        # temp workaround for indexing requirement of dataframe api
        self._index_columns = []

    @property
    def is_distributed(self):
        return self._is_distributed

    def distributed(self):
        self._is_distributed = True
        self._initialize_context()
        return self

    @property
    def context(self):
        return self._context

    def _initialize_context(self):
        if self.is_distributed:
            mpi_config = MPIConfig()
            self._context = CylonContext(config=mpi_config, distributed=True)
        else:
            self._context = CylonContext(config=None, distributed=False)

    def _initialize_dataframe(self, data=None, index=None, columns=None, copy=False):
        rows = 0
        cols = 0
        self._table = None
        if copy:
            data = self._copy(data)

        if isinstance(data, List):
            # load from List or np.ndarray
            if isinstance(data[0], List):
                rows = len(data[0])
                cols = len(data)
                if not columns:
                    columns = self._initialize_columns(
                        cols=cols, columns=columns)
                return cn.Table.from_list(self.context, columns, data)
            elif isinstance(data[0], np.ndarray):
                # load from List of np.ndarray
                cols = len(data)
                rows = data[0].shape[0]
                if not columns:
                    columns = self._initialize_columns(
                        cols=cols, columns=columns)
                return cn.Table.from_numpy(self.context, columns, data)
            else:
                # load from List
                rows = len(data)
                cols = 1
                if not columns:
                    columns = self._initialize_columns(
                        cols=cols, columns=columns)
                return cn.Table.from_list(self.context, columns, data)
        elif isinstance(data, pd.DataFrame):
            # load from pd.DataFrame
            rows, cols = data.shape
            if columns:
                from pycylon.util.pandas.utils import rename_with_new_column_names
                columns = rename_with_new_column_names(data, columns)
                data = data.rename(columns=columns, inplace=True)
            return cn.Table.from_pandas(self.context, data)
        elif isinstance(data, dict):
            # load from dictionary
            _, data_items = list(data.items())[0]
            rows = len(data_items)
            return cn.Table.from_pydict(self.context, data)
        elif isinstance(data, pa.Table):
            # load from pa.Table
            rows, cols = data.shape
            return cn.Table.from_arrow(self.context, data)
        elif isinstance(data, Series):
            # load from PyCylon Series
            # cols, rows = data.shape
            # columns = self._initialize_columns(cols=cols, columns=columns)
            return NotImplemented
        elif isinstance(data, cn.Table):
            if columns:
                from pycylon.util.pandas.utils import rename_with_new_column_names
                columns = rename_with_new_column_names(data, columns)
                data = data.rename(columns=columns)
            return data
        else:
            raise ValueError(f"Invalid data structure, {type(data)}")

    def _initialize_dtype(self, dtype):
        raise NotImplemented(
            "Data type forcing is not implemented, only support inferring types")

    def _initialize_columns(self, cols, columns):
        if columns is None:
            return [str(i) for i in range(cols)]
        else:
            if isinstance(columns, Iterable):
                if len(columns) != cols:
                    raise ValueError(f"data columns count: {cols} and column names count "
                                     f"{len(columns)} not equal")
                else:
                    return columns

    def _initialize_index(self, index, rows):
        if index is None:
            self._index = RangeIndex(start=0, stop=rows)
        else:
            if isinstance(index, CategoricalIndex):
                # check the validity of provided Index
                pass
            elif isinstance(index, RangeIndex):
                # check the validity of provided Index
                pass

    def _copy(self, obj):
        return copy(obj)

    @property
    def shape(self):
        return self._table.shape

    @property
    def columns(self) -> List[str]:
        return self._table.column_names

    def to_pandas(self) -> pd.DataFrame:
        return self._table.to_pandas()

    def to_numpy(self, order: str = 'F', zero_copy_only: bool = True, writable: bool = False) -> \
            np.ndarray:
        return self._table.to_numpy(self, order=order, zero_copy_only=zero_copy_only,
                                    writable=writable)

    def to_arrow(self) -> pa.Table:
        return self._table.to_arrow()

    def to_dict(self) -> Dict:
        return self._table.to_pydict()

    def to_table(self) -> cn.Table:
        return self._table

    def to_csv(self, path, csv_write_options: CSVWriteOptions):
        self._table.to_csv(path=path, csv_write_options=csv_write_options)

    def __getitem__(self, item) -> DataFrame:
        """
            This method allows to retrieve a subset of a DataFrane by means of a key
            Args:
                key: a key can be the following
                     1. slice i.e dataframe[1:5], rows 1:5
                     2. int i.e a row index
                     3. str i.e extract the data column-wise by column-name
                     4. List of columns are extracted
                     5. PyCylon DataFrame
            Returns: PyCylon DataFrame

            Examples
            --------
            >>> data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
            >>> df: DataFrame = DataFrame(data)

            >>> df1 = df[1:3]
                col-1  col-2  col-3
                    0      2      6     10
                    1      3      7     11
                    2      4      8     12

            >>> df2 = df['col-1']
                   col-1
                0      1
                1      2
                2      3
                3      4

            >>> df3 = df[['col-1', 'col-2']]
                   col-1  col-2
                0      1      5
                1      2      6
                2      3      7
                3      4      8

            >>> df4 = df > 3
                     col-1  col-2  col-3
                0    False   True   True
                1    False   True   True
                2    False   True   True
                3     True   True   True

            >>> df5 = df[tb4]
                    col-1  col-2  col-3
                0    NaN      5      9
                1    NaN      6     10
                2    NaN      7     11
                3    4.0      8     12

            >>> df8 = df['col-1'] > 2
                   col-1  col-2  col-3
                0      3      7     11
                1      4      8     12

        """
        if isinstance(item, slice) or isinstance(item, int) or isinstance(item, str) or \
                isinstance(item, List):
            return DataFrame(self._table.__getitem__(item))
        elif isinstance(item, DataFrame):
            return DataFrame(self._table.__getitem__(item.to_table()))

    def __setitem__(self, key, value):
        '''
            Sets values for a existing dataframe by means of a column
            Args:
                key: (str) column-name
                value: (DataFrame) data as a single column table

            Returns: PyCylon DataFrame

            Examples
            --------
            >>> df
                   col-1  col-2  col-3
                0      1      5      9
                1      2      6     10
                2      3      7     11
                3      4      8     12


            >>> df['col-3'] = DataFrame([[90, 100, 110, 120]])
                   col-1  col-2  col-3
                0      1      5     90
                1      2      6    100
                2      3      7    110
                3      4      8    120

            >>> df['col-4'] = DataFrame([190, 1100, 1110, 1120]])
                    col-1  col-2  col-3  col-4
                0      1      5     90    190
                1      2      6    100   1100
                2      3      7    110   1110
                3      4      8    120   1120
        '''

        if isinstance(key, str) and isinstance(value, DataFrame):
            self._table.__setitem__(key, value.to_table())
        else:
            raise ValueError(f"Not Implemented __setitem__ option for key Type {type(key)} and "
                             f"value type {type(value)}")

    def __repr__(self):
        return self._table.__repr__()

    def __eq__(self, other) -> DataFrame:
        '''
                Equal operator for DataFrame
                Args:
                    other: can be a numeric scalar or a DataFrame

                Returns: PyCylon DataFrame

                Examples
                --------

                >>> df
                       col-1  col-2  col-3
                    0      1      5      9
                    1      2      6     10
                    2      3      7     11
                    3      4      8     12

                >>> df['col-1'] == 2
                       col-1
                    0  False
                    1   True
                    2  False
                    3  False

                >>> df == 2
                       col-1  col-2  col-3
                    0  False  False  False
                    1   True  False  False
                    2  False  False  False
                    3  False  False  False

                '''
        return DataFrame(self._table.__eq__(other))

    def __ne__(self, other) -> DataFrame:
        '''
        Not equal operator for DataFrame
        Args:
            other: can be a numeric scalar or DataFrame

        Returns: PyCylon DataFrame

        Examples
        --------
        >>> df
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> df3 = df['col-1'] != 2
               col-1
            0   True
            1  False
            2   True
            3   True

        >>> df4 = df != 2
               col-1  col-2  col-3
            0   True   True   True
            1  False   True   True
            2   True   True   True
            3   True   True   True
        '''

        return DataFrame(self._table.__ne__(other))

    def __lt__(self, other) -> DataFrame:
        '''
        Lesser than operator for DataFrame
        Args:
            other: can be a numeric scalar or DataFrame

        Returns: PyCylon DataFrame

        Examples
        --------
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
        '''

        return DataFrame(self._table.__lt__(other))

    def __gt__(self, other) -> DataFrame:
        '''
        Greater than operator for DataFrame
        Args:
            other: can be a numeric scalar or DataFrame

        Returns: PyCylon DataFrame

        Examples
        --------
        >>> df
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> df3 = df['col-1'] > 2
                col-1
            0  False
            1  False
            2   True
            3   True

        >>> df4 = df > 2
               col-1  col-2  col-3
            0  False   True   True
            1  False   True   True
            2   True   True   True
            3   True   True   True
        '''

        return DataFrame(self._table.__gt__(other))

    def __le__(self, other) -> DataFrame:
        '''
        Lesser than or equal operator for DataFrame
        Args:
            other: can be a numeric scalar or DataFrame

        Returns: PyCylon DataFrame

        Examples
        --------
        >>> tb
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> df3 = df['col-1'] <= 2
                col-1
            0   True
            1   True
            2  False
            3  False

        >>> df4 = df <= 2
               col-1  col-2  col-3
            0   True  False  False
            1   True  False  False
            2  False  False  False
            3  False  False  False
        '''
        return DataFrame(self._table.__le__(other))

    def __ge__(self, other) -> DataFrame:
        '''
        Greater than or equal operator for DataFrame
        Args:
            other: can be a numeric scalar or DataFrame

        Returns: PyCylon DataFrame

        Examples
        --------
        >>> df
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12


        >>> df3 = df['col-1'] >= 2
               col-1
            0  False
            1   True
            2   True
            3   True

        >>> df4 = df >= 2
               col-1  col-2  col-3
            0  False   True   True
            1   True   True   True
            2   True   True   True
            3   True   True   True
        '''

        return DataFrame(self._table.__ge__(other))

    def __or__(self, other) -> DataFrame:
        '''
        Or operator for DataFrame
        Args:
            other: PyCylon DataFrame

        Returns: PyCylon DataFrame

        Examples
        --------
        >>> df1
               col-1  col-2
            0  False   True
            1   True   True
            2  False  False
            3   True  False

        >>> df2
                col-1  col-2
            0   True  False
            1   True   True
            2  False  False
            3  False   True

        >>> df_or = df1 | df2
               col-1  col-2
            0   True   True
            1   True   True
            2  False  False
            3   True   True
        '''

        return DataFrame(self._table.__or__(other.to_table()))

    def __and__(self, other) -> DataFrame:
        '''
        And operator for DataFrame
        Args:
            other: PyCylon DataFrame

        Returns: PyCylon DataFrame

        Examples
        --------
        >>> df1
               col-1  col-2
            0  False   True
            1   True   True
            2  False  False
            3   True  False

        >>> df2
                col-1  col-2
            0   True  False
            1   True   True
            2  False  False
            3  False   True

        >>> df_or = df1 & df2
               col-1  col-2
            0  False  False
            1   True   True
            2  False  False
            3  False  False
        '''

        return DataFrame(self._table.__and__(other.to_table()))

    def __invert__(self) -> DataFrame:
        '''
         Invert operator for DataFrame

         Returns: PyCylon DataFrame

         Examples
         --------
         >>> df
                col-1  col-2
            0  False   True
            1   True   True
            2  False  False
            3   True  False

        >>> ~df
               col-1  col-2
            0   True  False
            1  False  False
            2   True   True
            3  False   True
         '''

        return DataFrame(self._table.__invert__())

    def __neg__(self) -> DataFrame:
        '''
         Negation operator for DataFrame

         Returns: PyCylon DataFrame

         Examples
         --------
         >>> df
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

         >>> -df
               col-1  col-2  col-3
            0     -1     -5     -9
            1     -2     -6    -10
            2     -3     -7    -11
            3     -4     -8    -12
         '''

        return DataFrame(self._table.__neg__())

    def __add__(self, other) -> DataFrame:
        '''
         Add operator for DataFrame
         Args:
             other: scalar numeric

         Returns: PyCylon DataFrame

         Examples
         --------
        >>> df
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> df + 2
               col-1  col-2  col-3
            0      3      7     11
            1      4      8     12
            2      5      9     13
            3      6     10     14
         '''
        return DataFrame(self._table.__add__(other))

    def __sub__(self, other) -> DataFrame:
        '''
         Subtract operator for DataFrame
         Args:
             other: scalar numeric

         Returns: PyCylon DataFrame

         Examples
         --------
        >>> df
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> df - 2
               col-1  col-2  col-3
            0     -1      3      7
            1      0      4      8
            2      1      5      9
            3      2      6     10
         '''
        return DataFrame(self._table.__sub__(other))

    def __mul__(self, other) -> DataFrame:
        '''
         Multiply operator for DataFrame
         Args:
             other: scalar numeric

         Returns: PyCylon DataFrame

         Examples
         --------
        >>> df
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> df * 2
               col-1  col-2  col-3
            0      2     10     18
            1      4     12     20
            2      6     14     22
            3      8     16     24
         '''

        return DataFrame(self._table.__mul__(other))

    def __truediv__(self, other) -> DataFrame:
        '''
         Element-wise division operator for DataFrame
         Args:
             other: scalar numeric

         Returns: PyCylon DataFrame

         Examples
         --------
        >>> df
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> df / 2
               col-1  col-2  col-3
            0    0.5    2.5    4.5
            1    1.0    3.0    5.0
            2    1.5    3.5    5.5
            3    2.0    4.0    6.0
         '''

        return DataFrame(self._table.__truediv__(other))

    def drop(self, column_names: List[str]) -> DataFrame:
        '''
        drop a column or list of columns from a DataFrame
        Args:
            column_names: List[str]

        Returns: PyCylon DataFrame

        Examples
        --------

        >>> df
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> df.drop(['col-1'])
               col-2  col-3
            0      5      9
            1      6     10
            2      7     11
            3      8     12
        '''

        return DataFrame(self._table.drop(column_names))

    def fillna(self, fill_value) -> DataFrame:
        '''
        Fill not applicable values with a given value
        Args:
            fill_value: scalar

        Returns: PyCylon DataFrame

        Examples
        --------
        >>> df
               col-1  col-2  col-3
            0    1.0    5.0    9.0
            1    NaN    6.0   10.0
            2    3.0    NaN   11.0
            3    4.0    8.0    NaN

        >>> df.fillna(0)
               col-1  col-2  col-3
            0      1      5      9
            1      0      6     10
            2      3      0     11
            3      4      8      0
        '''
        # Note: Supports numeric types only
        return DataFrame(self._table.fillna(fill_value))

    def where(self, condition: DataFrame = None, other=None) -> DataFrame:
        '''
        Experimental version of Where operation.
        Replace values where condition is False
        Args:
            condition: bool DataFrame
            other: Scalar

        Returns: PyCylon DataFrame

        Examples
        --------
        >>> df
               col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> df.where(df > 2)
                col-1  col-2  col-3
            0    NaN      5      9
            1    NaN      6     10
            2    3.0      7     11
            3    4.0      8     12

        >>> df.where(df > 2, 10)
               col-1  col-2  col-3
            0     10      5      9
            1     10      6     10
            2      3      7     11
            3      4      8     12
        '''
        if condition is None:
            raise ValueError("Condition must be provided")
        return DataFrame(self._table.where(condition, other))

    def isnull(self) -> DataFrame:
        '''
        Checks for null elements and returns a bool DataFrame
        Returns: PyCylon DataFrame

        Examples
        --------

        >>> df
               col-1  col-2  col-3
            0    1.0    5.0    9.0
            1    NaN    6.0   10.0
            2    3.0    NaN   11.0
            3    4.0    8.0    NaN

        >>> df.isnull()
                col-1  col-2  col-3
            0  False  False  False
            1   True  False  False
            2  False   True  False
            3  False  False   True

        '''
        return DataFrame(self._table.isnull())

    def isna(self) -> DataFrame:
        '''
        Check for not applicable values and returns a bool DataFrame
        Returns: PyCylon DataFrame

        Examples
        --------
        >>> df
               col-1  col-2  col-3
            0    1.0    5.0    9.0
            1    NaN    6.0   10.0
            2    3.0    NaN   11.0
            3    4.0    8.0    NaN

        >>> df.isna()
                col-1  col-2  col-3
            0  False  False  False
            1   True  False  False
            2  False   True  False
            3  False  False   True
        '''
        return DataFrame(self._table.isnull())

    def notnull(self) -> DataFrame:
        '''
        Check the not null values and returns a bool DataFrame
        Returns: PyCylon DataFrame

        Examples
        --------
        >>> df
               col-1  col-2  col-3
            0    1.0    5.0    9.0
            1    NaN    6.0   10.0
            2    3.0    NaN   11.0
            3    4.0    8.0    NaN

        >>> df.notnull()
               col-1  col-2  col-3
            0   True   True   True
            1  False   True   True
            2   True  False   True
            3   True   True  False
        '''

        return ~self.isnull()

    def notna(self) -> DataFrame:
        '''
        Checks for not NA values and returns a bool DataFrame
        Returns: PyCylon DataFrame

        Examples
        --------
        >>> df
               col-1  col-2  col-3
            0    1.0    5.0    9.0
            1    NaN    6.0   10.0
            2    3.0    NaN   11.0
            3    4.0    8.0    NaN

        >>> df.notna()
               col-1  col-2  col-3
            0   True   True   True
            1  False   True   True
            2   True  False   True
            3   True   True  False
        '''

        return ~self.isnull()

    def rename(self, column_names):
        '''
        Rename a DataFrame with a column name or column names
        Args:
            column_names: dictionary or full list of new column names

        Returns: None

        Examples
        --------
        >>> df
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> df.rename({'col-1': 'col_1'})
               col_1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> df.rename(['c1', 'c2', 'c3'])
               c1  c2  c3
            0   1   5   9
            1   2   6  10
            2   3   7  11
            3   4   8  12
        '''
        self._table.rename(column_names)

    def add_prefix(self, prefix: str) -> DataFrame:
        '''
        Adding a prefix to column names
        Args:
            prefix: str

        Returns: PyCylon DataFrame with prefix updated

        Examples
        --------

        >>> df
                col-1  col-2  col-3
            0      1      5      9
            1      2      6     10
            2      3      7     11
            3      4      8     12

        >>> df.add_prefix('old_')
               old_c1  old_c2  old_c3
            0       1       5       9
            1       2       6      10
            2       3       7      11
            3       4       8      12

        '''

        return DataFrame(self._table.add_prefix(prefix))

    # Indexing

    def set_index(
        self, keys, drop=True, append=False, inplace=False, verify_integrity=False
    ):
        """
        Set the DataFrame index using existing columns.
        Set the DataFrame index (row labels) using one or more existing
        columns or arrays (of the correct length). The index can replace the
        existing index or expand on it.
        Parameters
        ----------
        keys : label or array-like or list of labels/arrays
            This parameter can be either a single column key, a single array of
            the same length as the calling DataFrame, or a list containing an
            arbitrary combination of column keys and arrays. Here, "array"
            encompasses :class:`Series`, :class:`Index`, ``np.ndarray``, and
            instances of :class:`~collections.abc.Iterator`.
        drop : bool, default True
            Delete columns to be used as the new index.
        append : bool, default False
            Whether to append columns to existing index.
        inplace : bool, default False
            If True, modifies the DataFrame in place (do not create a new object).
        verify_integrity : bool, default False
            Check the new index for duplicates. Otherwise defer the check until
            necessary. Setting to False will improve the performance of this
            method.
        Returns
        -------
        DataFrame or None
            Changed row labels or None if ``inplace=True``.
        See Also
        --------
        DataFrame.reset_index : Opposite of set_index.
        DataFrame.reindex : Change to new indices or expand indices.
        DataFrame.reindex_like : Change to same indices as other DataFrame.
        Examples
        --------
        >>> df = pd.DataFrame({'month': [1, 4, 7, 10],
        ...                    'year': [2012, 2014, 2013, 2014],
        ...                    'sale': [55, 40, 84, 31]})
        >>> df
           month  year  sale
        0      1  2012    55
        1      4  2014    40
        2      7  2013    84
        3     10  2014    31
        Set the index to become the 'month' column:
        >>> df.set_index('month')
               year  sale
        month
        1      2012    55
        4      2014    40
        7      2013    84
        10     2014    31
        Create a MultiIndex using columns 'year' and 'month':
        >>> df.set_index(['year', 'month'])
                    sale
        year  month
        2012  1     55
        2014  4     40
        2013  7     84
        2014  10    31
        Create a MultiIndex using an Index and a column:
        >>> df.set_index([pd.Index([1, 2, 3, 4]), 'year'])
                 month  sale
           year
        1  2012  1      55
        2  2014  4      40
        3  2013  7      84
        4  2014  10     31
        Create a MultiIndex using two Series:
        >>> s = pd.Series([1, 2, 3, 4])
        >>> df.set_index([s, s**2])
              month  year  sale
        1 1       1  2012    55
        2 4       4  2014    40
        3 9       7  2013    84
        4 16     10  2014    31
        """
        # todo this is not a final implementation
        self._index_columns = keys
        self._table.set_index(keys, drop=drop)
        return self

    def reset_index(  # type: ignore[misc]
        self,
        level: Optional[Union[Hashable, Sequence[Hashable]]] = ...,
        drop: bool = ...,
        inplace: Literal[False] = ...,
        col_level: Hashable = ...,
        col_fill=...,
    ) -> DataFrame:
        # todo this is not a final implementation
        self._index_columns = []
        self._table.reset_index(drop=drop)
        return self

    # Combining / joining / merging

    def join(self, other: DataFrame, on=None, how='left', lsuffix='', rsuffix='',
             sort=False, algorithm="sort"):
        """
        Join columns with other DataFrame either on index or on a key
        column. Efficiently Join multiple DataFrame objects by index at once by
        passing a list.
        Parameters
        ----------
        other : DataFrame, Series with name field set, or list of DataFrame
            Index should be similar to one of the columns in this one. If a
            Series is passed, its name attribute must be set, and that will be
            used as the column name in the resulting joined DataFrame
        on : column name, tuple/list of column names, or array-like
            Column(s) in the caller to join on the index in other,
            otherwise joins index-on-index. If multiples
            columns given, the passed DataFrame must have a MultiIndex. Can
            pass an array as the join key if not already contained in the
            calling DataFrame. Like an Excel VLOOKUP operation
        how : {'left', 'right', 'outer', 'inner'}, default: 'left'
            How to handle the operation of the two objects.
            * left: use calling frame's index (or column if on is specified)
            * right: use other frame's index
            * outer: form union of calling frame's index (or column if on is
              specified) with other frame's index, and sort it
              lexicographically
            * inner: form intersection of calling frame's index (or column if
              on is specified) with other frame's index, preserving the order
              of the calling's one
        lsuffix : string
            Suffix to use from left frame's overlapping columns
        rsuffix : string
            Suffix to use from right frame's overlapping columns
        sort : boolean, default False
            Order result DataFrame lexicographically by the join key. If False,
            the order of the join key depends on the join type (how keyword)
        algorithm: {'sort', 'hash'}, default: 'sort'
            The algorithm that should be used to perform the join between two tables.
        Notes
        -----
        on, lsuffix, and rsuffix options are not supported when passing a list
        of DataFrame objects
        Examples
        --------
        >>> caller
            A key
        0  A0  K0
        1  A1  K1
        2  A2  K2
        3  A3  K3
        4  A4  K4
        5  A5  K5

        >>> other
            B key
        0  B0  K0
        1  B1  K1
        2  B2  K2
        Join DataFrames using their indexes.
        >>> caller.join(other, lsuffix='_caller', rsuffix='_other')
        >>>     A key_caller    B key_other
            0  A0         K0   B0        K0
            1  A1         K1   B1        K1
            2  A2         K2   B2        K2
            3  A3         K3  NaN       NaN
            4  A4         K4  NaN       NaN
            5  A5         K5  NaN       NaN
        If we want to join using the key columns, we need to set key to be
        the index in both caller and other. The joined DataFrame will have
        key as its index.
        >>> caller.set_index('key').join(other.set_index('key'))
        >>>      A    B
            key
            K0   A0   B0
            K1   A1   B1
            K2   A2   B2
            K3   A3  NaN
            K4   A4  NaN
            K5   A5  NaN
        Another option to join using the key columns is to use the on
        parameter. DataFrame.join always uses other's index but we can use any
        column in the caller. This method preserves the original caller's
        index in the result.
        >>> caller.join(other.set_index('key'), on='key')
        >>>     A key    B
            0  A0  K0   B0
            1  A1  K1   B1
            2  A2  K2   B2
            3  A3  K3  NaN
            4  A4  K4  NaN
            5  A5  K5  NaN
        See also
        --------
        DataFrame.merge : For column(s)-on-columns(s) operations
        Returns
        -------
        joined : DataFrame
        """
        left_on = on
        if left_on is None:
            left_on = self._index_columns

        right_on = other._index_columns

        joined_table = self._table.join(table=other._table, join_type=how,
                                        algorithm=algorithm,
                                        left_on=left_on, right_on=right_on,
                                        left_prefix=lsuffix, right_prefix=rsuffix)
        return DataFrame(joined_table)
