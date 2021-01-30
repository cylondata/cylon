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
from libcpp.vector cimport vector
from libcpp.string cimport string
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
import numpy as np

from cpython.ref cimport PyObject
from cython.operator cimport dereference as deref, preincrement as inc

from typing import List


'''
Cylon Indexing is done with the following enums. 
'''
ctypedef long LONG
ctypedef double DOUBLE
ctypedef float FLOAT
ctypedef int INT

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


cdef vector[void*] _get_void_vec_from_pylist(py_list, arrow_type):
    if arrow_type == pa.int64():
        return _get_long_vector_from_pylist(py_list)


cdef vector[void*] _get_long_vector_from_pylist(py_list):
    cdef vector[long] vec = range(len(py_list))
    cdef vector[void*] void_vec

    cdef int i = 0
    cdef int j = 0
    #cdef void* element
    #cdef long c_element

    for _ in range(len(py_list)):
        print("Values: ", _, <long>py_list[j])
        vec[j] = <long>py_list[j]
        j = j + 1

    # cdef vector[long].iterator it = vec.begin()
    #
    # while it != vec.end():
    #     print("c_element: ", deref(it))
    #     void_vec.push_back(deref(it)) #it #it[0]
    #     inc(it)

    for _ in range(len(py_list)):
        print("long vec values, ", _, vec.at(i))
        void_vec.push_back(&vec.at(i))
        i = i + 1

    return void_vec

cdef long get_long_value_from_pyobject(py_object):
    cdef long cast_value = <long> py_object
    return cast_value


cdef class PyObjectToCObject:
    def __cinit__(self, arrow_type):
        self.c_ptr = NULL
        self._arrow_type = arrow_type

    cdef void *get_cptr_from_object(self, py_object):
        cdef long l_cval
        cdef void* c_ptr
        if (self._arrow_type == pa.int64()):
            print("condition match")
            l_cval = <long> py_object
            c_ptr = <void *> l_cval
        return c_ptr

    cdef bool to_bool(self, py_object):
        cdef bool res = <bool> py_object
        return res

    cdef signed char to_int8(self, py_object):
        cdef signed char res = <signed char> py_object
        return res

    cdef signed short to_int16(self, py_object):
        cdef signed short res = <signed short> py_object
        return res

    cdef signed int to_int32(self, py_object):
        cdef int res = <int> py_object
        return res

    cdef signed long long to_int64(self, py_object):
        cdef signed long long res = <signed long long> py_object
        return res

    cdef unsigned char to_uint8(self, py_object):
        cdef unsigned char res = <unsigned char> py_object
        return res

    cdef unsigned short to_uint16(self, py_object):
        cdef unsigned short res = <unsigned short> py_object
        return res

    cdef unsigned int to_uint32(self, py_object):
        cdef unsigned int res = <unsigned int> py_object
        return res

    cdef unsigned long long to_uint64(self, py_object):
        cdef unsigned long long res = <unsigned long long> py_object
        return res

    cdef long to_long(self, py_object):
        cdef long res = <long> py_object
        return res

    cdef float to_float(self, py_object):
        cdef float res = <float> py_object
        return res

    cdef double to_double(self, py_object):
        cdef double res = <double> py_object
        return res

    cdef unsigned short to_half_float(self, py_object):
        cdef unsigned short res = <unsigned short> py_object
        return res

    cdef string to_string(self, py_object):
        cdef string res = <string> py_object
        return res


    cdef vector[void*] get_vector_ptrs_from_list(self, py_list):

        cdef vector[void*] c_vec
        cdef void *c_ptr
        cdef PyObject obj

        return c_vec

cdef class LocIndexer:



    def __cinit__(self, CIndexingSchema indexing_schema):
        self.indexer_shd_ptr = make_shared[CLocIndexer](indexing_schema)

    def loc_with_multi_column(self, indices, column_list, table):
        pass

    def loc_with_column_range(self, indices, column_range, table):
        pass

    def loc_with_single_column(self, indices, column_index, table):

        cdef shared_ptr[CTable] output
        cdef void* c_start_index  #= <void*> &a
        cdef void* c_end_index
        cdef int c_column_index = <int> column_index
        cdef string cs_start_index
        cdef string cs_end_index
        cdef float cf_start_index
        cdef float cf_end_index
        cdef double cd_start_index
        cdef double cd_end_index

        index = table.get_index()
        arrow_type = index.get_index_array().type
        ctx = table.context

        cdef PyObjectToCObject p2c = PyObjectToCObject(arrow_type)
        cdef shared_ptr[CTable] input = pycylon_unwrap_table(table)
        intermediate_tables = []

        if isinstance(indices, List):
            print("select set of indices")

            for index in indices:
                if arrow_type == pa.bool_():
                    c_start_index = <void*> p2c.to_bool(index)
                elif arrow_type == pa.uint8():
                    c_start_index = <void*> p2c.to_uint8(index)
                elif arrow_type == pa.int8():
                    c_start_index = <void*> p2c.to_int8(index)
                elif arrow_type == pa.uint16():
                    c_start_index = <void*> p2c.to_uint16(index)
                elif arrow_type == pa.int16():
                    c_start_index = <void*> p2c.to_int16(index)
                elif arrow_type == pa.uint32():
                    c_start_index = <void*> p2c.to_uint32(index)
                elif arrow_type == pa.int32():
                    c_start_index = <void*> p2c.to_int32(index)
                elif arrow_type == pa.uint64():
                    c_start_index = <void*> p2c.to_uint64(index)
                elif arrow_type == pa.int64():
                    c_start_index = <void*>p2c.to_long(index)
                elif arrow_type == pa.float16():
                    c_start_index = <void*>p2c.to_half_float(index)
                elif arrow_type == pa.float32():
                    cf_start_index = p2c.to_float(index)
                    self.indexer_shd_ptr.get().loc(&cf_start_index, c_column_index, input, output)
                    intermediate_tables.append(pycylon_wrap_table(output))
                elif arrow_type == pa.float64():
                    cd_start_index = p2c.to_double(index)
                    self.indexer_shd_ptr.get().loc(&cd_start_index, c_column_index, input, output)
                    intermediate_tables.append(pycylon_wrap_table(output))
                elif arrow_type == pa.string():
                    cs_start_index = p2c.to_string(index)
                    self.indexer_shd_ptr.get().loc(&cs_start_index, c_column_index, input, output)
                    intermediate_tables.append(pycylon_wrap_table(output))
                else:
                    raise ValueError("Unsupported data type")

                if arrow_type != pa.string() and arrow_type != pa.float32() and arrow_type != \
                        pa.float64():
                    self.indexer_shd_ptr.get().loc(&c_start_index, c_column_index, input, output)
                    intermediate_tables.append(pycylon_wrap_table(output))

            return Table.merge(ctx, intermediate_tables)

        if np.isscalar(indices):
            print("select a single index")
            if arrow_type == pa.bool_():
                c_start_index = <void*> p2c.to_bool(indices)
            if arrow_type == pa.uint8():
                c_start_index = <void*> p2c.to_uint8(indices)
            if arrow_type == pa.int8():
                c_start_index = <void*> p2c.to_int8(indices)
            if arrow_type == pa.uint16():
                c_start_index = <void*> p2c.to_uint16(indices)
            if arrow_type == pa.int16():
                c_start_index = <void*> p2c.to_int16(indices)
            if arrow_type == pa.uint32():
                c_start_index = <void*> p2c.to_uint32(indices)
            if arrow_type == pa.int32():
                c_start_index = <void*> p2c.to_int32(indices)
            if arrow_type == pa.uint64():
                c_start_index = <void*> p2c.to_uint64(indices)
            if arrow_type == pa.int64():
                c_start_index = <void*> p2c.to_long(indices)
            if arrow_type == pa.float16():
                c_start_index = <void*> p2c.to_half_float(indices)
            if arrow_type == pa.float32():
                cf_start_index = p2c.to_float(indices)
                self.indexer_shd_ptr.get().loc(&cf_start_index, c_column_index, input, output)
            if arrow_type == pa.float64():
                cd_start_index = p2c.to_double(indices)
                self.indexer_shd_ptr.get().loc(&cd_start_index, c_column_index, input, output)
            if arrow_type == pa.string():
                cs_start_index = p2c.to_string(indices)
                self.indexer_shd_ptr.get().loc(&cs_start_index, c_column_index, input, output)
            if arrow_type != pa.string() and arrow_type != pa.float32() and arrow_type != \
                    pa.float64():
                self.indexer_shd_ptr.get().loc(&c_start_index, c_column_index, input, output)

            return pycylon_wrap_table(output)

        if isinstance(indices, slice):
            print("select a range of index")
            # assume step = 1
            # TODO: generalize for slice with multi-steps then resolve index list
            start_index = indices.start
            end_index = indices.stop
            if arrow_type == pa.bool_():
                c_start_index = <void*> p2c.to_bool(start_index)
                c_end_index = <void*> p2c.to_bool(end_index)
            if arrow_type == pa.uint8():
                c_start_index = <void*> p2c.to_uint8(start_index)
                c_end_index = <void*> p2c.to_uint8(end_index)
            if arrow_type == pa.int8():
                c_start_index = <void*> p2c.to_int8(start_index)
                c_end_index = <void*> p2c.to_int8(end_index)
            if arrow_type == pa.uint16():
                c_start_index = <void*> p2c.to_uint16(start_index)
                c_end_index = <void*> p2c.to_uint16(end_index)
            if arrow_type == pa.int16():
                c_start_index = <void*> p2c.to_int16(start_index)
                c_end_index = <void*> p2c.to_int16(end_index)
            if arrow_type == pa.uint32():
                c_start_index = <void*> p2c.to_uint32(start_index)
                c_end_index = <void*> p2c.to_uint32(end_index)
            if arrow_type == pa.int32():
                c_start_index = <void*> p2c.to_int32(start_index)
                c_end_index = <void*> p2c.to_int32(end_index)
            if arrow_type == pa.uint64():
                c_start_index = <void*> p2c.to_uint64(start_index)
                c_end_index = <void*> p2c.to_uint64(end_index)
            if arrow_type == pa.int64():
                c_start_index = <void*> p2c.to_long(start_index)
                c_end_index = <void*> p2c.to_long(end_index)
            if arrow_type == pa.float16():
                c_start_index = <void*> p2c.to_half_float(start_index)
                c_end_index = <void*> p2c.to_half_float(end_index)
            if arrow_type == pa.float32():
                cf_start_index = p2c.to_float(start_index)
                cf_end_index = p2c.to_float(end_index)
                self.indexer_shd_ptr.get().loc(&cf_start_index, &cf_end_index, c_column_index,
                                               input, output)
            if arrow_type == pa.float64():
                cd_start_index = p2c.to_double(start_index)
                cd_end_index = p2c.to_double(end_index)
                self.indexer_shd_ptr.get().loc(&cd_start_index, &cd_end_index, c_column_index, \
                                                                             input, output)
            if arrow_type == pa.string():
                cs_start_index = p2c.to_string(start_index)
                cs_end_index = p2c.to_string(end_index)
                self.indexer_shd_ptr.get().loc(&cs_start_index, &cs_end_index, c_column_index,
                                               input, output)

            if arrow_type != pa.string() and arrow_type != pa.float32() and arrow_type != \
                    pa.float64():
                self.indexer_shd_ptr.get().loc(&c_start_index, &c_end_index, c_column_index,
                                               input, output)
            return pycylon_wrap_table(output)

