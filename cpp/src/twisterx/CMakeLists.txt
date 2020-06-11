add_library(twisterx SHARED
        net/ops/all_to_all.cpp net/ops/all_to_all.hpp
        net/channel.hpp
        net/mpi/mpi_channel.hpp net/mpi/mpi_channel.cpp
        net/mpi/mpi_communicator.h net/mpi/mpi_communicator.cpp
        arrow/arrow_all_to_all.cpp arrow/arrow_all_to_all.hpp
        join/join.hpp join/join.cpp
        util/arrow_utils.hpp util/arrow_utils.cpp
        arrow/arrow_kernels.cpp arrow/arrow_kernels.hpp
        util/copy_arrray.cpp join/join_utils.hpp
        join/join_utils.cpp arrow/arrow_join.cpp
        arrow/arrow_join.hpp util/sort_indices.cpp
        io/arrow_io.cpp io/arrow_io.hpp
        table_api.hpp table_api.cpp
        status.hpp
        util/to_string.hpp
        table.cpp table.hpp
        column.cpp column.hpp
        data_types.hpp
        arrow/arrow_types.hpp arrow/arrow_types.cpp
        arrow/arrow_partition_kernels.hpp arrow/arrow_partition_kernels.cpp
        util/murmur3.cpp util/murmur3.hpp
        join/join_config.h
        io/csv_read_config.h io/csv_read_config.cpp
        io/csv_read_config_holder.hpp
        util/uuid.hpp util/uuid.cpp
        net/TxRequest.h net/TxRequest.cpp
        util/builtins.h util/builtins.cpp
        ctx/twisterx_context.h ctx/twisterx_context.cpp table_api_extended.hpp
        io/csv_write_config.h io/csv_write_config.cpp
        python/ctx/twisterx_context_wrap.h python/ctx/twisterx_context_wrap.cpp
        arrow/arrow_hash_kernels.hpp
        arrow/arrow_comparator.h arrow/arrow_comparator.cpp
        row.hpp row.cpp)

include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(twisterx ${MPI_LIBRARIES})
target_link_libraries(twisterx glog::glog)
target_link_libraries(twisterx ${ARROW_LIB})
target_link_libraries(twisterx ${PYTHON_LIBRARIES})

if(${PYTWISTERX_BUILD})

add_library(twisterx_python SHARED
        python/table_cython.h python/table_cython.cpp
        python/net/distributed.h python/net/distributed.cpp
        python/net/comm/all_to_all_wrap.h python/net/comm/all_to_all_wrap.cpp
        python/net/comm/callback.h python/net/comm/callback.cpp
        python/ctx/twisterx_context_wrap.h python/ctx/twisterx_context_wrap.cpp)

target_link_libraries(twisterx_python ${MPI_LIBRARIES})
target_link_libraries(twisterx_python glog::glog)
target_link_libraries(twisterx_python ${ARROW_LIB})
target_link_libraries(twisterx_python ${ARROW_PYTHON})
target_link_libraries(twisterx_python ${PYTHON_LIBRARIES})
target_link_libraries(twisterx_python twisterx)

else()
    message("Skip Building TwisterX Cython-CPP")
endif()
