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

IF CYTHON_GLOO:
    from libcpp.memory cimport shared_ptr
    from libcpp.string cimport string
    from mpi4py.libmpi cimport MPI_Comm

    from pycylon.net.comm_type cimport CCommType
    from pycylon.net.comm_config cimport CommConfig, CCommConfig

    cdef extern from "../../../../cpp/src/cylon/net/gloo/gloo_communicator.hpp" namespace "cylon::net":
        cdef cppclass CGlooConfig "cylon::net::GlooConfig":
            int rank()
            int world_size()
            CCommType Type()

            void SetTcpHostname(const string& tcp_hostname)
            void SetTcpIface(const string & tcp_iface)
            void SetTcpAiFamily(int tcp_ai_family)
            void SetFileStorePath(const string & file_store_path)
            void SetStorePrefix(const string & store_prefix)

            @staticmethod
            shared_ptr[CGlooConfig] MakeWithMpi(MPI_Comm comm);

            @staticmethod
            shared_ptr[CGlooConfig] Make(int rank, int world_size);


    cdef class GlooMPIConfig(CommConfig):
        cdef:
            shared_ptr[CGlooConfig] gloo_config_shd_ptr

    cdef class GlooStandaloneConfig(CommConfig):
        cdef:
            shared_ptr[CGlooConfig] gloo_config_shd_ptr

ELSE:
    cdef class GlooMPIConfig(CommConfig):
        pass

    cdef class GlooStandaloneConfig(CommConfig):
        pass
