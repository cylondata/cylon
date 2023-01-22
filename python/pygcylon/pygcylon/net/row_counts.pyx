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
from libcpp.vector cimport vector
from pycylon.api.lib cimport pycylon_unwrap_context

from pygcylon.net.row_counts cimport RowCountsAllTables


def row_counts_all_tables(num_rows, context):
    cdef CStatus status
    cdef shared_ptr[CCylonContext] c_ctx_ptr = pycylon_unwrap_context(context)
    cdef vector[int] c_all_table_rows

    status =  RowCountsAllTables(num_rows=num_rows, ctx_srd_ptr=c_ctx_ptr, all_num_rows=c_all_table_rows)

    if status.is_ok():
        return c_all_table_rows
    else:
        raise Exception(f"NumRowsAllTables operation failed : {status.get_msg().decode()}")

