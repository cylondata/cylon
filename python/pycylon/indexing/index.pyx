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
from pycylon.indexing.index cimport CLocIndexer
from pycylon.indexing.index cimport CBaseIndex
from pyarrow.lib cimport (pyarrow_unwrap_table, pyarrow_wrap_table, pyarrow_wrap_array,
pyarrow_unwrap_array)

from pycylon.api.lib cimport (pycylon_wrap_context, pycylon_unwrap_context, pycylon_unwrap_table,
pycylon_wrap_table)

from pycylon.data.table cimport CTable
from pycylon.data.table import Table
from pycylon.ctx.context cimport CCylonContext
from pycylon.ctx.context import CylonContext
import pyarrow as pa

from cpython.ref cimport PyObject

'''
Cylon Indexing is done with the following enums. 
'''

cpdef enum IndexingSchema:
    RANGE = CIndexingSchema.CRANGE
    LINEAR = CIndexingSchema.CLINEAR
    HASH = CIndexingSchema.CHASH
    BINARYTREE = CIndexingSchema.CBINARYTREE
    BTREE = CIndexingSchema.CBTREE

cdef class BaseIndex:
    cdef void init(self, const shared_ptr[CBaseIndex]& index):
        self.bindex_shd_ptr = index

    def get_index_array(self) -> pa.array:
        cdef shared_ptr[CArrowArray] index_arr = self.bindex_shd_ptr.get().GetIndexArray()
        py_arw_index_arr = pyarrow_wrap_array(index_arr)
        return py_arw_index_arr

cdef class PyObjectToCObject:

    def __cinit__(self, py_object, arrow_type):
        cdef long l_cval
        if(arrow_type == pa.int64()):
            print("condition match")
            l_cval = <long> py_object
            self.c_ptr = <void *> l_cval

    cdef void * get_ptr(self):
        return self.c_ptr

cdef class LocIndexer:
    def __cinit__(self, CIndexingSchema indexing_schema):
        self.indexer_shd_ptr = make_shared[CLocIndexer](indexing_schema)

    cdef void resolve_ctype_from_python_object(self, py_object, index,  void *c_ptr):
        print("resolve_ctype_from_python_object")
        index_arr = index.get_index_array()
        cdef long c_value
        cdef void * cptr
        cdef long * res
        if (index_arr.type == pa.int64()):
            print("long selected")
            c_value = <long> py_object
            print("casted value : ", c_value)
            cptr = <void *> &c_value

    def loc(self, start_index, end_index, column_index, table):
        cdef shared_ptr[CTable] output
        cdef long a = <long>start_index
        cdef long b = <long>end_index
        cdef void* c_start_index #= <void*> &a
        cdef void* c_end_index #= <void*> &b
        cdef int c_column_index = <int> column_index
        index = table.get_index()

        #self.resolve_ctype_from_python_object(start_index, index, c_start_index)
        #self.resolve_ctype_from_python_object(end_index, index, c_end_index)
        cdef PyObjectToCObject p2c_s = PyObjectToCObject(start_index, index.get_index_array().type)
        cdef PyObjectToCObject p2c_e = PyObjectToCObject(end_index, index.get_index_array().type)
        cdef shared_ptr[CTable] input = pycylon_unwrap_table(table)

        c_start_index = p2c_s.get_ptr()
        c_end_index = p2c_e.get_ptr()

        self.indexer_shd_ptr.get().loc(&c_start_index, &c_end_index, c_column_index, input, output)
        return pycylon_wrap_table(output)
