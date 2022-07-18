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
    from pycylon.net.comm_config cimport CommConfig
    from pycylon.net.gloo_config cimport CGlooConfig
    cimport mpi4py.MPI as MPI

    from mpi4py.MPI import COMM_NULL

    cdef class GlooMPIConfig(CommConfig):
        """
        GlooConfig Type mapping from libCylon to PyCylon
        """
        def __cinit__(self, comm = COMM_NULL):
            self.gloo_config_shd_ptr = CGlooConfig.MakeWithMpi((<MPI.Comm> comm).ob_mpi)

        @property
        def rank(self):
            return self.gloo_config_shd_ptr.get().rank()

        @property
        def world_size(self):
            return self.gloo_config_shd_ptr.get().world_size()

        @property
        def comm_type(self):
            return self.gloo_config_shd_ptr.get().Type()

        def set_tcp_hostname(self, hostname: str):
            self.gloo_config_shd_ptr.get().SetTcpHostname(hostname.encode())

        def set_tcp_iface(self, iface: str):
            self.gloo_config_shd_ptr.get().SetTcpIface(iface.encode())

        def set_tcp_ai_family(self, ai_family: int):
            self.gloo_config_shd_ptr.get().SetTcpAiFamily(ai_family)

    cdef class GlooStandaloneConfig(CommConfig):
        """
        GlooConfig Type mapping from libCylon to PyCylon
        """
        def __cinit__(self, rank = 0, world_size = 1):
            if rank < 0 or world_size < 0:
                raise ValueError(f"Invalid rank/ world size provided")
            self.gloo_config_shd_ptr = CGlooConfig.Make(rank, world_size)

        @property
        def rank(self):
            return self.gloo_config_shd_ptr.get().rank()

        @property
        def world_size(self):
            return self.gloo_config_shd_ptr.get().world_size()

        @property
        def comm_type(self):
            return self.gloo_config_shd_ptr.get().Type()

        def set_tcp_hostname(self, hostname: str):
            self.gloo_config_shd_ptr.get().SetTcpHostname(hostname.encode())

        def set_tcp_iface(self, iface: str):
            self.gloo_config_shd_ptr.get().SetTcpIface(iface.encode())

        def set_tcp_ai_family(self, ai_family: int):
            self.gloo_config_shd_ptr.get().SetTcpAiFamily(ai_family)

        def set_file_store_path(self, path: str):
            self.gloo_config_shd_ptr.get().SetFileStorePath(path.encode())

        def set_store_prefix(self, prefix: str):
            self.gloo_config_shd_ptr.get().SetStorePrefix(prefix.encode())
