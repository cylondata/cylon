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
from pyarrow.lib cimport *


cdef class ArrowUtil:

    @staticmethod
    def get_array_length(obj):
        '''

        :param obj: passing a PyArrow array
        :return: length from the shared pointer for the corresponding python object
        '''
        # Just an example function accessing both the pyarrow Cython API
        # and the Arrow C++ API
        cdef shared_ptr[CArray] arr = pyarrow_unwrap_array(obj)
        if arr.get() == NULL:
            raise TypeError("not an array")
        return arr.get().length()

    @staticmethod
    def get_array_info(obj):
        '''
        Test method to get information from extracted C++ shared object
        :param obj: Pyarrow array
        :return: None
        '''
        # Just an example function accessing both the pyarrow Cython API
        # and the Arrow C++ API
        cdef shared_ptr[CArray] arr = pyarrow_unwrap_array(obj)
        if arr.get() == NULL:
            raise TypeError("not an array")
        else:
            print("Length : {}".format(arr.get().length()))
            print("Num Fields : {}".format(arr.get().num_fields()))

    @staticmethod
    def get_table_info(obj):
        '''
        Test method for getting table info from shared pointer
        :param obj: PyArrow Table
        :return: None
        '''
        cdef shared_ptr[CTable] artb = pyarrow_unwrap_table(obj)

        if artb.get() == NULL:
            raise TypeError("not an table")
        else:
            print("OK")
            num_cols = artb.get().num_columns()
            print(num_cols)


