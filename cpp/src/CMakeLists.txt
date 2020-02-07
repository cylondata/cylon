add_executable(twisterx config/Config.h config/Config.cpp all_to_all.cpp all_to_all.hpp channel.hpp mpi_channel.hpp request.hpp type.hpp mpi_channel.cpp)
add_library(twisterxlib all_to_all.cpp all_to_all.hpp channel.hpp mpi_channel.hpp request.hpp type.hpp mpi_channel.cpp arrow/arrow_all_to_all.cpp arrow/arrow_all_to_all.hpp)

if(MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    target_link_libraries(twisterx ${MPI_LIBRARIES})
    target_link_libraries(twisterxlib ${MPI_LIBRARIES})
    target_link_libraries(twisterxlib glog::glog)
    target_link_libraries(twisterx glog::glog)
    target_link_libraries(twisterx ${ARROW_LIBRARIES})
endif()


