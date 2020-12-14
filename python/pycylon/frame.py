from __future__ import annotations
from typing import List, Dict
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
        self._table = self._initialize_dataframe(data=data, index=index, columns=columns, copy=copy)

    @property
    def is_distributed(self):
        return self._is_distributed

    def distributed(self):
        self._is_distributed = True
        self._initialize_context()

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
                    columns = self._initialize_columns(cols=cols, columns=columns)
                return cn.Table.from_list(self.context, columns, data)
            elif isinstance(data[0], np.ndarray):
                # load from List of np.ndarray
                cols = len(data)
                rows = data[0].shape[0]
                if not columns:
                    columns = self._initialize_columns(cols=cols, columns=columns)
                return cn.Table.from_numpy(self.context, columns, data)
            else:
                # load from List
                rows = len(data)
                cols = 1
                if not columns:
                    columns = self._initialize_columns(cols=cols, columns=columns)
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
        raise NotImplemented("Data type forcing is not implemented, only support inferring types")

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

