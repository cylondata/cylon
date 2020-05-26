from libcpp.vector cimport vector

cdef extern from "../../../cpp/src/twisterx/python/net/comm/all_to_all_wrap.h" namespace "twisterx::net::comm":
    cdef cppclass CAll_to_all_wrap "twisterx::net::comm::all_to_all_wrap":
        CAll_to_all_wrap();
        CAll_to_all_wrap(int worker_id, const vector[int] &source, const vector[int] &targets, int edgeId);
        void insert(void *buffer, int length, int target, int *header, int headerLength);
        void wait();
        void finish();