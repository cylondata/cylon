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

from pycylon.io.csv_read_config cimport CCSVReadOptions
from libcpp cimport bool
from typing import List
from libcpp.string cimport string
from libcpp.vector cimport vector

'''
This API is work in progress 
We currenly support data loading through PyArrow APIs with minimum overheads and full compatibility with
Pandas. 
'''

cdef class CSVReadOptions:

    def __cinit__(self):
        self.thisPtr = new CCSVReadOptions()

    def use_threads(self, use_thread: bool)-> CSVReadOptions:
        self.thisPtr.UseThreads(use_thread)
        return self

    def block_size(self, block_size: int) -> CSVReadOptions:
        self.thisPtr.BlockSize(block_size)
        return self

    def with_delimiter(self, delimiter: str) -> CSVReadOptions:
        cdef char c_delimiter = delimiter.encode()[0]
        self.thisPtr.WithDelimiter(c_delimiter)
        return self

    def ignore_emptylines(self) -> CSVReadOptions:
        self.thisPtr.IgnoreEmptyLines()
        return self


