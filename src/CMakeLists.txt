include_directories(../include)

add_executable(twisterx twisterx.cpp api/worker/Worker.h api/config/Config.h api/config/Config.cpp api/comm/Communicator.h api/comm/Types.h api/comm/Buffer.h api/comm/packers/DataPacker.h api/comm/packers/PackerStatus.h api/comm/messages/OutMessage.h api/comm/messages/Message.h api/comm/op/GatherTest.h api/comm/Receiver.h all_to_all.cpp ../include/all_to_all.hpp ../include/channel.hpp ../include/mpi_channel.hpp ../include/request.hpp ../include/type.hpp  mpi_channel.hpp)

if(MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    target_link_libraries(twisterx ${MPI_LIBRARIES})
endif()
