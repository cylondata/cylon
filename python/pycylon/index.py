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


from pycylon.data.table import Table
from typing import List
from numbers import Number
import warnings
import numpy as np
from pycylon.util.TableUtils import resolve_column_index_from_column_name
from pycylon.indexing.index import IndexingSchema, BaseIndex
from pycylon.indexing.index_utils import IndexUtil


class Index(object):

    def __init__(self, data):
        pass

    def initialize(self):
        pass

    @property
    def index(self):
        return self


class NumericIndex(Index):
    # TODO: Extend Index Utils and Functions: https://github.com/cylondata/cylon/issues/230
    def __init__(self, data):
        self._index_values = data
        self.initialize()

    def initialize(self):
        pass

    @property
    def index_values(self):
        return self._index_values

    @index_values.setter
    def index_values(self, data):
        self._index_values = data


class IntegerIndex(NumericIndex):
    # TODO: Extend Index Utils and Functions: https://github.com/cylondata/cylon/issues/230
    def __init__(self, data):
        super().__init__(data=data)

    def initialize(self):
        super().initialize()


class RangeIndex(IntegerIndex):
    # TODO: Extend Index Utils and Functions: https://github.com/cylondata/cylon/issues/230
    def __init__(self, data=None, start: int = 0, stop: int = 0, step: int = 0):
        self._start = start
        self._stop = stop
        self._step = step
        super().__init__(data=data)

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, start):
        self._start = start

    @property
    def stop(self):
        return self._stop

    @stop.setter
    def stop(self, stop):
        self._stop = stop

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step):
        self._step = step

    def initialize(self):
        if self.stop == 0:
            if isinstance(self._index_values, range):
                self.start = self._index_values.start
                self.stop = self._index_values.stop
                self.step = self._index_values.step
            else:
                warnings.warn("Empty Range!. Range data or range criteria must be passed")
        else:
            self._index_values = range(self.start, self.stop, self.step)


class CategoricalIndex(Index):
    # TODO: Extend Index Utils and Functions: https://github.com/cylondata/cylon/issues/230
    def __init__(self, key):
        self._index = key

    @property
    def index_values(self):
        return self._index


class ColumnIndex(Index):
    # TODO: Extend Index Utils and Functions: https://github.com/cylondata/cylon/issues/230
    def __init__(self, key):
        self._index = key

    @property
    def index_values(self):
        return self._index


def range_calculator(rg: range):
    start = rg.start
    step = rg.step
    stop = rg.stop
    if step != 1:
        diff = stop - start
        length = diff // step
        if diff % step == 0:
            return length
        else:
            return length + 1
    else:
        return stop - start


def _is_index_and_range_validity(table, index_range):
    if isinstance(index_range, range):
        return range_calculator(index_range) == table.row_count
    else:
        return False


def _is_index_list_and_valid(table, index):
    if isinstance(index, List):
        if len(index) == table.row_count:
            return True
        else:
            return False
    else:
        return False


def _is_index_list_of_columns(table, index):
    if isinstance(index, List):
        for index_item in index:
            if index_item not in table.column_names:
                return False
        return True
    else:
        return False


def _get_index_list_from_columns(table, index):
    # multi-column indexing
    index_columns = []
    ar_tb = table.to_arrow().combine_chunks()
    if isinstance(index, List):
        for index_item in index:
            index_columns.append(ar_tb.column(index_item))
        return index_columns
    elif isinstance(index, str):
        index_columns.append(ar_tb.column(index))
        return index_columns
    else:
        return NotImplemented("Not Supported index pattern")


def _is_index_str_and_valid(table, index):
    if isinstance(index, str):
        if index in table.column_names:
            return True
    return False


def _get_column_by_name(table, column_name):
    artb = table.to_arrow().combine_chunks()
    return artb.column(column_name)


def process_index_by_value(key, table, index_schema, drop_index, method="arrow"):
    if np.isscalar(key):
        column_index = None
        if isinstance(key, str):
            column_index = resolve_column_index_from_column_name(key, table)
        elif isinstance(key, int):
            column_index = key
        if method == "default":
            return IndexUtil.build_index(index_schema, table, column_index, drop_index)
        elif method == "arrow":
            return IndexUtil.build_arrow_index(index_schema, table, column_index, drop_index)
    elif isinstance(key, List):
        if method == "default":
            return IndexUtil.build_index_from_list(index_schema, table, key)
        elif method == "arrow":
            return IndexUtil.build_arrow_index_from_list(index_schema, table, key)
    else:
        raise ValueError("Unexpected value")


def _process_index_by_value(key, table):
    # TODO: Indexing Python enhancement for Indexing Dev Stage 2
    index = None
    if _is_index_list_and_valid(table, key):
        # scenario-1: key is a list of row_indices
        index = CategoricalIndex(key)
    elif _is_index_list_of_columns(table, key):
        # scenario-2: key is a list of columns in the table
        index = ColumnIndex(_get_index_list_from_columns(table, key))
    elif isinstance(key, range):
        range_index = RangeIndex(key)
        process_index_by_value(range_index, table)
    elif isinstance(key, CategoricalIndex):
        if _is_index_list_and_valid(table, key.index_values):
            index = key
    elif isinstance(key, NumericIndex) or isinstance(key, RangeIndex):
        # scenario-3 key is a Index which can be a Numerical or RangeIndex
        if _is_index_list_and_valid(table, key.index_values):
            index = CategoricalIndex(key)
        elif _is_index_and_range_validity(table, key.index_values):
            index = RangeIndex(key)
        else:
            raise ValueError(f"Index type {key.index_values} not supported or invalid!")
    elif _is_index_str_and_valid(table, key):
        # scenario-4: key is a column name in the existing table
        index = ColumnIndex(_get_column_by_name(table, key))
    else:
        raise ValueError(f"Index type {key} not supported or invalid!")
    return index
