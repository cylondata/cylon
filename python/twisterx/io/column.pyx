from twisterx.io.column cimport Column
from libcpp.string cimport string

cdef class PyColumn:

    cdef Column *thisptr

    def __cinit__(self, string id):
        self.thisptr = new Column(id)

    def __dealloc__(self):
        del self.thisptr

    cpdef get_id(self):
        return self.thisptr.get_id()