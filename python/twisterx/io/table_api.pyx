from libcpp.memory cimport shared_ptr, make_shared
from libcpp.string cimport string

from twisterx.io.table_api cimport table_api

cdef class PyTableAPI:
    cdef table_api *thisptr

    def __cinit__(self, string id):
        self.thisptr = new table_api()

    def __dealloc__(self):
        del self.thisptr

    def read_csv(self, string path, string id ):
        self.thisptr.read_csv(path, id)