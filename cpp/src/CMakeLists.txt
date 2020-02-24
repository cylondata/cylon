add_library(twisterx SHARED all_to_all.cpp all_to_all.hpp channel.hpp mpi_channel.hpp request.hpp type.hpp mpi_channel.cpp arrow/arrow_all_to_all.cpp arrow/arrow_all_to_all.hpp join/tx_join.hpp join/tx_join.cpp util/arrow_utils.h util/arrow_utils.cpp)

if(MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    target_link_libraries(twisterx ${MPI_LIBRARIES})
    target_link_libraries(twisterx glog::glog)
endif()


