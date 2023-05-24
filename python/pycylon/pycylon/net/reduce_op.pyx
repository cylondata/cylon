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
Mapping Cylon C++ Comm Type with PyCylon CommType
'''

from pycylon.net.reduce_op cimport CReduceOp

cpdef enum ReduceOp:
    SUM = CReduceOp._SUM
    MIN = CReduceOp._MIN
    MAX = CReduceOp._MAX
    PROD = CReduceOp._PROD
    LAND = CReduceOp._LAND
    LOR = CReduceOp._LOR
    BAND = CReduceOp._BAND
    BOR = CReduceOp._BOR