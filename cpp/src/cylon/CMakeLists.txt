add_library(cylon SHARED
        net/ops/all_to_all.cpp net/ops/all_to_all.hpp
        net/channel.hpp
        net/mpi/mpi_channel.hpp net/mpi/mpi_channel.cpp
        net/mpi/mpi_communicator.hpp net/mpi/mpi_communicator.cpp
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
        join/join_config.hpp
        io/csv_read_config.hpp io/csv_read_config.cpp
        io/csv_read_config_holder.hpp
        util/uuid.hpp util/uuid.cpp
        net/TxRequest.hpp net/TxRequest.cpp
        util/builtins.hpp util/builtins.cpp
        ctx/cylon_context.hpp ctx/cylon_context.cpp table_api_extended.hpp
        io/csv_write_config.hpp io/csv_write_config.cpp
        arrow/arrow_hash_kernels.hpp
        arrow/arrow_comparator.hpp arrow/arrow_comparator.cpp
        row.hpp row.cpp ctx/memory_pool.hpp ctx/arrow_memory_pool_utils.hpp ctx/arrow_memory_pool_utils.cpp)

include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(cylon ${MPI_LIBRARIES})
target_link_libraries(cylon glog::glog)
target_link_libraries(cylon ${ARROW_LIB})
target_link_libraries(cylon ${PYTHON_LIBRARIES})

if(${PYCYLON_BUILD})

add_library(cylon_python SHARED
        python/cylon_context_wrap.h python/cylon_context_wrap.cpp
        python/table_cython.h python/table_cython.cpp
        python/net/distributed.h python/net/distributed.cpp
        python/net/comm/all_to_all_wrap.h python/net/comm/all_to_all_wrap.cpp
        python/net/comm/callback.h python/net/comm/callback.cpp)

target_link_libraries(cylon_python ${MPI_LIBRARIES})
target_link_libraries(cylon_python glog::glog)
target_link_libraries(cylon_python ${ARROW_LIB})
target_link_libraries(cylon_python ${ARROW_PYTHON})
target_link_libraries(cylon_python ${PYTHON_LIBRARIES})
target_link_libraries(cylon_python cylon)

else()
    message("Skip Building Cylon Cython-CPP")
endif()
