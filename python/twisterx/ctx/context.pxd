# from libcpp.vector cimport vector
# from libcpp.string cimport string
#
# cdef extern from "../../../cpp/src/twisterx/ctx/twister_context.h" namespace "twisterx":
#     cdef cppclass CTwisterXContext "twisterx::twister_context":
#         CTwisterXContext *Init();
#         void Finalize();
#         #CTwisterXContext *InitDistributed(net::CommConfig *config);
#         void AddConfig(const string &key, const string &value);
#         string GetConfig(const string &key, const string &defn);
#         #net::Communicator *GetCommunicator() const;
#         int GetRank();
#         int GetWorldSize();
#         vector[int] GetNeighbours(bool include_self);