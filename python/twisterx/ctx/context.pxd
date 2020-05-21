from libcpp.vector cimport vector
from libcpp.string cimport string
#
cdef extern from "../../../cpp/src/twisterx/ctx/twisterx_context.h" namespace "twisterx":
    cdef cppclass CTwisterXContext "twisterx::twisterx_context":
        pass
#         void Finalize();
#         #CTwisterXContext *InitDistributed(net::CommConfig *config);
#         void AddConfig(const string &key, const string &value);
#         string GetConfig(const string &key, const string &defn);
#         #net::Communicator *GetCommunicator() const;
#         int GetRank();
#         int GetWorldSize();
#         vector[int] GetNeighbours(bool include_self);


cdef extern from "../../../cpp/src/twisterx/python/ctx/twisterx_context_wrap.h" namespace "twisterx::py":
    cdef cppclass CTwisterXContextWrap "twisterx::py::twisterx_context_wrap":
        CTwisterXContextWrap()
        CTwisterXContextWrap(string config)
        CTwisterXContext *getInstance()
        int GetRank()
        int GetWorldSize()
        void Finalize()