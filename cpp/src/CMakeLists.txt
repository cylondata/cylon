add_library(twisterx SHARED all_to_all.cpp all_to_all.hpp channel.hpp mpi_channel.hpp mpi_channel.cpp arrow/arrow_all_to_all.cpp arrow/arrow_all_to_all.hpp join/join.hpp join/join.cpp util/arrow_utils.hpp util/arrow_utils.cpp arrow/arrow_kernels.cpp arrow/arrow_kernels.hpp util/copy_arrray.cpp join/join_utils.hpp join/join_utils.cpp arrow/arrow_join.cpp arrow/arrow_join.hpp util/sort_indices.cpp)

include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(twisterx ${MPI_LIBRARIES})
target_link_libraries(twisterx glog::glog)


