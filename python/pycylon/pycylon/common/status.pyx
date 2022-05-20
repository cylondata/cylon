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
from pycylon.common.code cimport CCode
from pycylon.common.status cimport CStatus


cdef class Status:
    def __cinit__(self, CCode code=CCode._OK, string msg=b''):
        """
        Initializes the Status to wrap the C++ object in Cython
        :param msg: passes a str to convery the status message
        :param code: Cython correpondence to C++ Code object
        :return: None
        """
        if msg.empty():
            self.thisptr = new CStatus(code)
        else:
            self.thisptr = new CStatus(code, msg)

    def __del__(self):
        del self.thisptr

    def get_code(self):
        """

        :return: the code
        """
        return self.thisptr.get_code()

    def is_ok(self):
        """

        :return: OK status from Status
        """
        return self.thisptr.is_ok()

    def get_msg(self):
        """

        :return: Message from Status
        """
        return self.thisptr.get_msg().decode()
