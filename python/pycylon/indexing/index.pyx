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

from libcpp.memory cimport shared_ptr, make_shared
from pyarrow.lib cimport CArray as CArrowArray
from pycylon.indexing.index cimport CIndexingSchema
from pycylon.indexing.index cimport CBaseIndex
from pyarrow.lib cimport (pyarrow_unwrap_table, pyarrow_wrap_table, pyarrow_wrap_array)

from pycylon.api.lib cimport (pycylon_wrap_context, pycylon_unwrap_context, pycylon_unwrap_table,
                                pycylon_wrap_table)

'''
Cylon Indexing is done with the following enums. 
'''

cpdef enum IndexingSchema:
    RANGE = CIndexingSchema.CRANGE
    LINEAR = CIndexingSchema.CLINEAR
    HASH = CIndexingSchema.CHASH
    BINARYTREE = CIndexingSchema.CBINARYTREE
    BTREE = CIndexingSchema.CBTREE


