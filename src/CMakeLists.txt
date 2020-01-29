add_executable(twisterx twisterx.cpp api/worker/Worker.h config/Config.h config/Config.cpp api/comm/Communicator.h api/comm/Types.h api/comm/Buffer.h api/comm/packers/DataPacker.h api/comm/packers/PackerStatus.h api/comm/messages/OutMessage.h api/comm/messages/Message.h api/comm/op/GatherTest.h api/comm/Receiver.h all_to_all.cpp all_to_all.hpp channel.hpp mpi_channel.hpp request.hpp type.hpp mpi_channel.cpp)
add_library(twisterxlib twisterx.cpp api/worker/Worker.h config/Config.h config/Config.cpp api/comm/Communicator.h api/comm/Types.h api/comm/Buffer.h api/comm/packers/DataPacker.h api/comm/packers/PackerStatus.h api/comm/messages/OutMessage.h api/comm/messages/Message.h api/comm/op/GatherTest.h api/comm/Receiver.h all_to_all.cpp all_to_all.hpp channel.hpp mpi_channel.hpp request.hpp type.hpp mpi_channel.cpp)

if(MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    target_link_libraries(twisterx ${MPI_LIBRARIES})
    target_link_libraries(twisterxlib ${MPI_LIBRARIES})
    target_link_libraries(twisterxlib glog::glog)
    target_link_libraries(twisterx glog::glog)
    target_link_libraries(twisterx ${ARROW_LIBRARIES})
endif()


