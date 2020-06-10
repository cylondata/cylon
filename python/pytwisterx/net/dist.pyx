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
Wrapping for custom communication APIs for PyTwisterX
(Work in Progress)
'''

cdef extern from "../../../cpp/src/twisterx/python/net/distributed.h" namespace "twisterx::net":
    cdef extern void cdist_init()
    cdef extern void cdist_finalize()
    cdef extern int cget_rank()
    cdef extern int cget_size()


def dist_init():
    cdist_init()

def dist_finalize():
    cdist_finalize()

def rank() -> int:
    return cget_rank()

def size() -> int:
    return cget_size()