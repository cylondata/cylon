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
Cython Interface for Shuffle
'''

from libcpp.memory cimport shared_ptr
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pycylon.ctx.context cimport CCylonContext
from pycylon.common.status cimport CStatus

from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.table.table cimport table

#
cdef extern from "gcylon/gtable.hpp" namespace "gcylon":

    CStatus Shuffle(const table_view & inputTview,
                    const vector[int] & columns_to_hash,
                    shared_ptr[CCylonContext] ctx_srd_ptr,
                    unique_ptr[table] & outputTable);

