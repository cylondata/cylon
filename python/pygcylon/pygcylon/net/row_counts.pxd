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

'''
Cython Interface for RowCounts
'''

from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from pycylon.ctx.context cimport CCylonContext
from pycylon.common.status cimport CStatus

#
cdef extern from "gcylon/gtable_api.hpp" namespace "gcylon":

    CStatus RowCountsAllTables(int num_rows,
                               const shared_ptr[CCylonContext] & ctx_srd_ptr,
                               const vector[int] & all_num_rows);

