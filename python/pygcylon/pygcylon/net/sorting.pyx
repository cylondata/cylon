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

from libcpp.memory cimport shared_ptr
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool

from pycylon.ctx.context cimport CCylonContext
from pycylon.api.lib cimport pycylon_unwrap_context
from pygcylon.net.sorting cimport DistributedSort

from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.table cimport Table
cimport cudf._lib.cpp.types as libcudf_types


def distributed_sort(
        object tbl,
        context,
        object sort_columns=None,
        ascending=True,
        nulls_after=True,
        ignore_index=False,
        by_index=False,
):
    cdef Table c_table = tbl
    cdef table_view c_tv = c_table.data_view() if ignore_index else c_table.view()
    cdef vector[int] c_sort_column_indices
    cdef vector[libcudf_types.order] c_column_orders
    cdef CStatus status
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)
    cdef cppbool c_ascending = ascending
    cdef cppbool c_nulls_after = nulls_after

    # Determine sort_column indices from index names
    # ref: cudf/python/cudf/merge.pyx
    if ignore_index:
        num_index_columns = 0
        index_names = None
    else:
        num_index_columns = (
            0 if tbl._index is None
            else tbl._index._num_columns
        )
        index_names = tbl._index_names

    # put indices into the C vector
    if not by_index:
        num_indices = len(sort_columns)
        c_sort_column_indices.reserve(num_indices)
        for cname in sort_columns:
            c_sort_column_indices.push_back(
                num_index_columns + tbl._column_names.index(cname)
            )
    else:
        c_sort_column_indices.reserve(num_index_columns)
        for key in range(0, num_index_columns):
            c_sort_column_indices.push_back(key)

    # construct c_column_orders
    # ascending can be either a bool or a list of bools
    c_column_orders.reserve(c_sort_column_indices.size())
    if isinstance(ascending, list):
        for ascend in ascending:
            order = libcudf_types.order.ASCENDING if ascend else libcudf_types.order.DESCENDING
            c_column_orders.push_back(order)
    elif isinstance(ascending, bool):
        for _ in range(c_sort_column_indices.size()):
            order = libcudf_types.order.ASCENDING if ascending else libcudf_types.order.DESCENDING
            c_column_orders.push_back(order)
    else:
        raise ValueError("ascending must be either a bool or a list of bool")

    # Perform sorting
    cdef unique_ptr[table] c_sorted_table
#    with nogil:
    status = DistributedSort(
        c_tv,
        c_sort_column_indices,
        c_column_orders,
        c_ctx_ptr,
        c_sorted_table,
        c_nulls_after
    )

    if status.is_ok():
        return Table.from_unique_ptr(
            move(c_sorted_table),
            column_names=tbl._column_names,
            index_names=index_names,
        )
    else:
        raise ValueError(f"Sort operation failed : {status.get_msg().decode()}")
