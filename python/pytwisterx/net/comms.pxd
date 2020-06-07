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
libTwisterX mapping for All To ALl Communication.
'''

from libcpp.vector cimport vector

cdef extern from "../../../cpp/src/twisterx/python/net/comm/all_to_all_wrap.h" namespace "twisterx::net::comm":
    cdef cppclass CAll_to_all_wrap "twisterx::net::comm::all_to_all_wrap":
        CAll_to_all_wrap();
        CAll_to_all_wrap(int worker_id, const vector[int] &source, const vector[int] &targets, int edgeId);
        void insert(void *buffer, int length, int target, int *header, int headerLength);
        void wait();
        void finish();