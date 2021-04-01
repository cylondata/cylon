# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

find_package(Threads REQUIRED)

set_source_files_properties(util/murmur3.cpp util/murmur3.hpp PROPERTIES COMPILE_FLAGS -Wno-implicit-fallthrough)
if (CYLON_PARQUET)
    set(PARQUET_CYLON_FILES
            io/parquet_config.hpp
            io/parquet_config.cpp
            )
else (CYLON_PARQUET)
    set(PARQUET_CYLON_FILES)
endif (CYLON_PARQUET)

add_library(cylon SHARED
        net/ops/all_to_all.cpp
        net/ops/all_to_all.hpp
        net/channel.hpp
        net/mpi/mpi_channel.hpp
        net/mpi/mpi_channel.cpp
        net/mpi/mpi_communicator.hpp
        net/mpi/mpi_communicator.cpp
        arrow/arrow_all_to_all.cpp
        arrow/arrow_all_to_all.hpp
        join/join.hpp
        join/join.cpp
        join/hash_join.hpp
        join/hash_join.cpp
        join/sort_join.hpp
        join/sort_join.cpp
        util/arrow_utils.hpp
        util/arrow_utils.cpp
        arrow/arrow_kernels.cpp
        arrow/arrow_kernels.hpp
        util/copy_arrray.cpp
        join/join_utils.hpp
        join/join_utils.cpp
        io/arrow_io.cpp
        io/arrow_io.hpp
        table_api.hpp
        table_api.cpp
        status.hpp
        util/to_string.hpp
        table.cpp
        table.hpp
        column.cpp
        column.hpp
        data_types.hpp
        arrow/arrow_types.hpp
        arrow/arrow_types.cpp
        arrow/arrow_partition_kernels.hpp
        arrow/arrow_partition_kernels.cpp
        util/murmur3.cpp
        util/murmur3.hpp
        join/join_config.hpp
        io/csv_read_config.hpp
        io/csv_read_config.cpp
        io/csv_read_config_holder.hpp
        util/uuid.hpp
        util/uuid.cpp
        util/sort.hpp
        net/TxRequest.hpp
        net/TxRequest.cpp
        util/builtins.hpp
        util/builtins.cpp
        ctx/cylon_context.hpp
        ctx/cylon_context.cpp
        table_api_extended.hpp
        io/csv_write_config.hpp
        io/csv_write_config.cpp
        arrow/arrow_comparator.hpp
        arrow/arrow_comparator.cpp
        row.hpp
        row.cpp
        ctx/memory_pool.hpp
        ctx/arrow_memory_pool_utils.hpp
        ctx/arrow_memory_pool_utils.cpp
        arrow/arrow_builder.hpp
        arrow/arrow_builder.cpp
        arrow/arrow_task_all_to_all.h
        arrow/arrow_task_all_to_all.cpp
        ops/api/parallel_op.hpp
        ops/join_op.hpp
        ops/api/parallel_op.cpp
        ops/join_op.cpp
        ops/partition_op.hpp
        ops/partition_op.cpp
        ops/kernels/partition.hpp
        ops/kernels/partition.cpp
        ops/union_op.hpp
        ops/union_op.cpp
        ops/kernels/union.cpp
        ops/kernels/union.hpp
        ops/kernels/row_comparator.cpp
        ops/kernels/row_comparator.hpp
        ops/kernels/prepare_array.hpp
        ops/kernels/prepare_array.cpp
        ops/all_to_all_op.hpp
        ops/all_to_all_op.cpp
        ops/dis_union_op.hpp
        ops/dis_union_op.cpp
        ops/dis_join_op.cpp
        ops/dis_join_op.hpp
        ops/execution/execution.hpp
        ops/execution/execution.cpp
        ops/kernels/join_kernel.hpp
        ops/kernels/join_kernel.cpp
        ops/merge_op.cpp
        ops/merge_op.hpp
        ops/split_op.cpp
        ops/split_op.hpp
        arrow/arrow_builder.cpp
        compute/aggregates.hpp
        compute/aggregates.cpp
        compute/aggregate_utils.hpp
        compute/aggregate_kernels.hpp
        net/comm_operations.hpp
        net/mpi/mpi_operations.cpp
        net/mpi/mpi_operations.hpp
        groupby/groupby.hpp
        groupby/groupby.cpp
        groupby/pipeline_groupby.hpp
        groupby/pipeline_groupby.cpp
        groupby/hash_groupby.hpp
        groupby/hash_groupby.cpp
        ${PARQUET_CYLON_FILES}
        partition/partition.hpp
        partition/partition.cpp
        util/macros.hpp
        thridparty/flat_hash_map/unordered_map.hpp
        thridparty/flat_hash_map/flat_hash_map.hpp
        thridparty/flat_hash_map/bytell_hash_map.hpp
        indexing/index.hpp
        indexing/index.cpp
        indexing/index_utils.hpp
        indexing/index_utils.cpp
        indexing/indexer.hpp
        indexing/indexer.cpp
        )

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(cylon ${MPI_LIBRARIES})
target_link_libraries(cylon ${GLOG_LIBRARIES})
target_link_libraries(cylon ${ARROW_LIB})
target_link_libraries(cylon ${PYTHON_LIBRARIES})
target_link_libraries(cylon Threads::Threads)
target_compile_options(cylon PRIVATE -Werror -Wall -Wextra -Wno-unused-parameter)

if (CYLON_PARQUET)
    target_link_libraries(cylon ${PARQUET_LIB})
endif ()

cylon_install_all_headers("cylon")

add_subdirectory(util)
add_subdirectory(net)
add_subdirectory(join)
add_subdirectory(io)
add_subdirectory(ctx)
add_subdirectory(arrow)

set_target_properties(cylon PROPERTIES VERSION ${CYLON_VERSION})
install(TARGETS cylon DESTINATION lib)