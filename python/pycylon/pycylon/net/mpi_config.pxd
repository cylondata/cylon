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

from libcpp.memory cimport shared_ptr
from pycylon.net.comm_type cimport CCommType
from pycylon.net.comm_config cimport CCommConfig
from pycylon.net.comm_config import CommConfig
from pycylon.net.comm_config cimport CommConfig


cdef extern from "../../../../cpp/src/cylon/net/mpi/mpi_communicator.hpp" namespace "cylon::net":
    cdef cppclass CMPIConfig "cylon::net::MPIConfig":

        @staticmethod
        shared_ptr[CMPIConfig] Make();


cdef class MPIConfig(CommConfig):
    cdef:
        shared_ptr[CMPIConfig] mpi_config_shd_ptr