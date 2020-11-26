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

from libcpp.string cimport string
from libcpp cimport bool
from pycylon.common.status cimport CStatus
from pycylon.common.status import Status
from pycylon.io.csv_read_config cimport CCSVReadOptions
from pyarrow.lib cimport CTable as CArrowTable
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
from pycylon.data.table cimport CTable
from pycylon.data.table cimport Table
from pycylon.data.table import Table
from pycylon.api.lib cimport (pycylon_unwrap_context,
pycylon_wrap_table,
pycylon_unwrap_csv_read_options)


def read_csv(context, path, csv_read_options) -> Table:
        '''
            loading data from a csv file
            :param context: CylonContext
            :param path: Path to csv file
            :param csv_read_options: CSVReadOptions object
        '''
        cdef shared_ptr[CCylonContext] ctx = pycylon_unwrap_context(context)
        cdef string cpath = path.encode()
        cdef CCSVReadOptions c_csv_read_options = pycylon_unwrap_csv_read_options(csv_read_options)
        cdef shared_ptr[CTable] cn_table
        cdef CStatus status = FromCSV(ctx, cpath, cn_table, c_csv_read_options)
        if status.is_ok():
            return pycylon_wrap_table(cn_table)
        else:
            raise Exception(f"Table couldn't be created from CSV: {status.get_msg().decode()}")