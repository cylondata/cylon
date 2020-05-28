from libcpp.string cimport string
from libcpp cimport bool
from pytwisterx.common.code cimport _Code
from pytwisterx.common.status cimport _Status


cdef class Status:
    cdef _Status *thisptr
    cdef _Code _code
    cdef string msg
    cdef int code

    def __cinit__(self, int code, string msg, _Code _code):
        if _code != -1 and msg.size() == 0 and code == -1:
            #print("Status(_Code)")
            self.thisptr = new _Status(_code)
            self._code = _code

        if msg.size() != 0 and code != -1:
            #print("Status(code, msg)")
            self.thisptr = new _Status(code, msg)
            self.msg = msg
            self.code = code

        if msg.size() == 0 and _code == -1 and code != -1:
            #print("Status(code)")
            self.thisptr = new _Status(code)
            self.code = code

        if msg.size() != 0 and _code != -1 and code == -1:
            #print("Status(_Code, msg)")
            self.thisptr = new _Status(_code, msg)
            self._code = _code
            self.msg = msg

    def get_code(self):
        return self.thisptr.get_code()

    def is_ok(self):
        return self.thisptr.is_ok()

    def get_msg(self):
        return self.thisptr.get_msg()
