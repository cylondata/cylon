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
            #cols, rows = data.shape
            #columns = self._initialize_columns(cols=cols, columns=columns)
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

    def __getitem__(self, item):
        return DataFrame(self._table.__getitem__(item))

    def __setitem__(self, key, value):
        if isinstance(key, str) and isinstance(value, DataFrame):
            self._table.__setitem__(key, value.to_table())

    def __repr__(self):
        return self._table.__repr__()
