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

from pycylon.ctx.context cimport CCylonContext
from pycylon.api.lib cimport pycylon_unwrap_context
from pygcylon.net.repartition cimport Repartition

from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.table cimport Table


def repartition(
        object tbl,
        context,
        object rows_per_worker=None,
        ignore_index=False,
):
    cdef Table c_table = tbl
    cdef table_view c_tv = c_table.data_view() if ignore_index else c_table.view()
    cdef vector[int] c_rows_per_worker
    cdef CStatus status
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)

    if rows_per_worker:
        c_rows_per_worker = rows_per_worker

    index_names = None if ignore_index else tbl._index_names

    # Perform repartitioning
    cdef unique_ptr[table] c_table_out
    status = Repartition(
        c_tv,
        c_ctx_ptr,
        c_table_out,
        c_rows_per_worker
    )

    if status.is_ok():
        return Table.from_unique_ptr(
            move(c_table_out),
            column_names=tbl._column_names,
            index_names=index_names,
        )
    else:
        raise ValueError(f"Repartition operation failed : {status.get_msg().decode()}")
