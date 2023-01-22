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

import cudf
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.utils cimport data_from_unique_ptr, table_view_from_table


from pycylon.ctx.context cimport CCylonContext
from pycylon.api.lib cimport pycylon_unwrap_context
from pygcylon.net.c_comms cimport Repartition, Gather, Broadcast, AllGather

def repartition(object input_table, context, object rows_per_worker=None,
                ignore_index=False) -> cudf.DataFrame:
    cdef table_view c_tv = table_view_from_table(input_table, ignore_index=ignore_index)
    cdef vector[int] c_rows_per_worker
    cdef CStatus status
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)

    if rows_per_worker:
        c_rows_per_worker = rows_per_worker

    index_names = None if ignore_index else input_table._index_names

    # Perform repartitioning
    cdef unique_ptr[table] c_table_out
    status = Repartition(
        c_tv,
        c_ctx_ptr,
        c_table_out,
        c_rows_per_worker
    )

    if status.is_ok():
        return cudf.DataFrame._from_data(*data_from_unique_ptr(
            move(c_table_out), column_names=input_table._column_names, index_names=index_names))
    else:
        raise ValueError(f"Repartition operation failed : {status.get_msg().decode()}")

def gather(object input_table, context, object gather_root,
           ignore_index=False, ) -> cudf.DataFrame:
    cdef table_view c_tv = table_view_from_table(input_table, ignore_index=ignore_index)
    cdef int c_gather_root = gather_root
    cdef CStatus c_status
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)

    index_names = None if ignore_index else input_table._index_names

    # Perform repartitioning
    cdef unique_ptr[table] c_table_out
    c_status = Gather(
        c_tv,
        c_ctx_ptr,
        c_table_out,
        c_gather_root,
    )

    if c_status.is_ok():
        return cudf.DataFrame._from_data(*data_from_unique_ptr(
            move(c_table_out),
            column_names=input_table._column_names,
            index_names=index_names,
        ))
    else:
        raise ValueError(f"Gather operation failed : {c_status.get_msg().decode()}")

def allgather(object input_table, context,
              ignore_index=False, ) -> cudf.DataFrame:
    cdef table_view c_tv = table_view_from_table(input_table, ignore_index=ignore_index)
    cdef CStatus c_status
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)

    index_names = None if ignore_index else input_table._index_names

    # Perform repartitioning
    cdef unique_ptr[table] c_table_out
    c_status = AllGather(
        c_tv,
        c_ctx_ptr,
        c_table_out,
    )

    if c_status.is_ok():
        return cudf.DataFrame._from_data(*data_from_unique_ptr(
            move(c_table_out),
            column_names=input_table._column_names,
            index_names=index_names,
        ))
    else:
        raise ValueError(f"AllGather operation failed : {c_status.get_msg().decode()}")

def broadcast(object input_table, context, object root,
              ignore_index=False, ) -> cudf.DataFrame:
    cdef table_view c_tv = table_view_from_table(input_table, ignore_index=ignore_index)
    cdef int c_root = root
    cdef CStatus c_status
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)

    index_names = None if ignore_index else input_table._index_names

    # Perform repartitioning
    cdef unique_ptr[table] c_table_out
    c_status = Broadcast(
        c_tv,
        c_root,
        c_ctx_ptr,
        c_table_out,
    )

    if c_status.is_ok():
        return cudf.DataFrame._from_data(*data_from_unique_ptr(
            move(c_table_out),
            column_names=input_table._column_names,
            index_names=index_names,
        ))
    else:
        raise ValueError(f"Broadcast operation failed : {c_status.get_msg().decode()}")
