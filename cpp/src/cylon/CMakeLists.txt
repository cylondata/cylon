##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

find_package(Threads REQUIRED)

IF(WIN32)
    set_source_files_properties(util/murmur3.cpp util/murmur3.hpp PROPERTIES)
ELSE()
    set_source_files_properties(util/murmur3.cpp util/murmur3.hpp PROPERTIES COMPILE_FLAGS -Wno-implicit-fallthrough)
ENDIF()
if (CYLON_PARQUET)
    set(PARQUET_CYLON_FILES
            io/parquet_config.hpp
            io/parquet_config.cpp
            )
else (CYLON_PARQUET)
    set(PARQUET_CYLON_FILES)
endif (CYLON_PARQUET)

if (CYLON_UCX)
    set(UCX_CYLON_FILES
            net/ucx/ucx_channel.hpp
            net/ucx/ucx_channel.cpp
            net/ucx/ucx_communicator.hpp
            net/ucx/ucx_communicator.cpp
            net/ucx/ucx_operations.hpp
            net/ucx/ucx_operations.cpp
            )
else (CYLON_UCX)
    set(UCX_CYLON_FILES)
endif (CYLON_UCX)

add_library(cylon SHARED
        ${PARQUET_CYLON_FILES}
        ${UCX_CYLON_FILES}
        arrow/arrow_all_to_all.cpp
        arrow/arrow_all_to_all.hpp
        arrow/arrow_builder.cpp
        arrow/arrow_builder.cpp
        arrow/arrow_builder.hpp
        arrow/arrow_comparator.cpp
        arrow/arrow_comparator.hpp
        arrow/arrow_kernels.cpp
        arrow/arrow_kernels.hpp
        arrow/arrow_partition_kernels.cpp
        arrow/arrow_partition_kernels.hpp
        arrow/arrow_task_all_to_all.cpp
        arrow/arrow_task_all_to_all.h
        arrow/arrow_type_traits.hpp
        arrow/arrow_types.cpp
        arrow/arrow_types.hpp
        column.cpp
        column.hpp
        compute/aggregate_kernels.hpp
        compute/aggregate_utils.hpp
        compute/aggregates.cpp
        compute/aggregates.hpp
        ctx/arrow_memory_pool_utils.cpp
        ctx/arrow_memory_pool_utils.hpp
        ctx/cylon_context.cpp
        ctx/cylon_context.hpp
        ctx/memory_pool.hpp
        data_types.hpp
        groupby/groupby.cpp
        groupby/groupby.hpp
        groupby/hash_groupby.cpp
        groupby/hash_groupby.hpp
        groupby/pipeline_groupby.cpp
        groupby/pipeline_groupby.hpp
        indexing/index.cpp
        indexing/index.hpp
        indexing/index_utils.hpp
        indexing/index_utils.cpp
        indexing/indexer.cpp
        indexing/indexer.hpp
        io/arrow_io.cpp
        io/arrow_io.hpp
        io/csv_read_config.cpp
        io/csv_read_config.hpp
        io/csv_read_config_holder.hpp
        io/csv_write_config.cpp
        io/csv_write_config.hpp
        join/hash_join.cpp
        join/hash_join.hpp
        join/join.cpp
        join/join.hpp
        join/join_config.hpp
        join/join_utils.cpp
        join/join_utils.hpp
        join/sort_join.cpp
        join/sort_join.hpp
        net/channel.hpp
        net/comm_operations.hpp
        net/comm_type.hpp
        net/communicator.hpp
        net/mpi/mpi_channel.cpp
        net/mpi/mpi_channel.hpp
        net/mpi/mpi_communicator.cpp
        net/mpi/mpi_communicator.hpp
        net/mpi/mpi_operations.cpp
        net/mpi/mpi_operations.hpp
        net/mpi/mpi_type_traits.hpp
        net/ops/all_to_all.cpp
        net/ops/all_to_all.hpp
        net/ops/gather.cpp
        net/ops/bcast.cpp
        net/TxRequest.cpp
        net/TxRequest.hpp
        ops.cpp
        ops.hpp
        ops/all_to_all_op.cpp
        ops/all_to_all_op.hpp
        ops/api/parallel_op.cpp
        ops/api/parallel_op.hpp
        ops/dis_join_op.cpp
        ops/dis_join_op.hpp
        ops/dis_set_op.cpp
        ops/dis_set_op.hpp
        ops/execution/execution.cpp
        ops/execution/execution.hpp
        ops/join_op.cpp
        ops/join_op.hpp
        ops/kernels/join_kernel.cpp
        ops/kernels/join_kernel.hpp
        ops/kernels/partition.cpp
        ops/kernels/partition.hpp
        ops/kernels/prepare_array.cpp
        ops/kernels/prepare_array.hpp
        ops/kernels/row_comparator.cpp
        ops/kernels/row_comparator.hpp
        ops/kernels/set_kernel.cpp
        ops/kernels/set_kernel.hpp
        ops/merge_op.cpp
        ops/merge_op.hpp
        ops/partition_op.cpp
        ops/partition_op.hpp
        ops/set_op.cpp
        ops/set_op.hpp
        ops/split_op.cpp
        ops/split_op.hpp
        partition/partition.cpp
        partition/partition.hpp
        row.cpp
        row.hpp
        status.hpp
        table.cpp
        table.hpp
        table_api.cpp
        table_api.hpp
        table_api_extended.hpp
        repartition.hpp
        thridparty/flat_hash_map/bytell_hash_map.hpp
        thridparty/flat_hash_map/flat_hash_map.hpp
        thridparty/flat_hash_map/unordered_map.hpp
        util/arrow_rand.hpp
        util/arrow_utils.cpp
        util/arrow_utils.hpp
        util/builtins.cpp
        util/builtins.hpp
        util/copy_arrray.cpp
        util/flatten_array.cpp
        util/flatten_array.hpp
        util/logging.cpp
        util/logging.hpp
        util/macros.hpp
        util/murmur3.cpp
        util/murmur3.hpp
        util/sort.hpp
        util/to_string.hpp
        util/uuid.cpp
        util/uuid.hpp
        )

IF(NOT MSVC)
    if(APPLE)
        set(CMAKE_SHARED_LINKER_FLAGS "-Wl")
    else()
        set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
    endif()
ENDIF()

# link external libraries
target_link_libraries(cylon ${MPI_CXX_LIBRARIES})
target_link_libraries(cylon ${GLOG_LIBRARIES})
target_link_libraries(cylon ${ARROW_LIB})
target_link_libraries(cylon Threads::Threads)

if (CYLON_PARQUET)
    target_link_libraries(cylon ${PARQUET_LIB})
endif ()

if (CYLON_UCX)
    target_link_libraries(cylon ${UCX_LIBRARIES})
endif ()

if (PYCYLON_BUILD)
    target_link_libraries(cylon ${PYTHON_LIBRARIES})
    target_link_libraries(cylon ${ARROW_PY_LIB})
endif (PYCYLON_BUILD)

target_compile_options(cylon PRIVATE)
# target_compile_options(cylon PRIVATE -Werror -Wall -Wextra -Wno-unused-parameter)

cylon_install_all_headers("cylon")

add_subdirectory(util)
add_subdirectory(net)
add_subdirectory(join)
add_subdirectory(io)
add_subdirectory(ctx)
add_subdirectory(arrow)
add_subdirectory(indexing)
add_subdirectory(thridparty/flat_hash_map)
add_subdirectory(compute)

set_target_properties(cylon PROPERTIES VERSION ${CYLON_VERSION})

install(TARGETS cylon DESTINATION lib)