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

if (CYLON_BENCH)
    message("Benchmark timings enabled")
    add_definitions(-D_CYLON_BENCH)
endif ()

if (WIN32)
    set_source_files_properties(util/murmur3.cpp util/murmur3.hpp PROPERTIES)
else ()
    set_source_files_properties(util/murmur3.cpp util/murmur3.hpp PROPERTIES COMPILE_FLAGS -Wno-implicit-fallthrough)
endif (WIN32)

if (CYLON_UCX)
    set(UCX_CYLON_FILES
            net/ucx/ucx_channel.hpp
            net/ucx/ucx_channel.cpp
            net/ucx/ucx_communicator.hpp
            net/ucx/ucx_communicator.cpp
            net/ucx/ucx_operations.hpp
            net/ucx/ucx_operations.cpp
            net/ucx/mpi_ucx_ucc_oob_context.hpp
            net/ucx/mpi_ucx_ucc_oob_context.cpp
            )
    if (CYLON_USE_REDIS)
        set(UCX_REDIS_CYLON_FILES
                net/ucx/redis_ucx_ucc_oob_context.hpp
                net/ucx/redis_ucx_ucc_oob_context.cpp
                )
    endif(CYLON_USE_REDIS)
    if (CYLON_UCC)
        set(UCC_CYLON_FILES
                net/ucc/ucc_operations.hpp
                net/ucc/ucc_operations.cpp)
    else ()
        set(UCC_CYLON_FILES)
    endif (CYLON_UCC)
else (CYLON_UCX)
    set(UCX_CYLON_FILES)
    set(UCC_CYLON_FILES)
endif (CYLON_UCX)

if (CYLON_GLOO)
    set(CYLON_GLOO_FILES
            net/gloo/gloo_channel.cpp
            net/gloo/gloo_channel.hpp
            net/gloo/gloo_communicator.cpp
            net/gloo/gloo_communicator.hpp
            net/gloo/gloo_operations.cpp
            net/gloo/gloo_operations.hpp)
else (CYLON_GLOO)
    set(CYLON_GLOO_FILES)
endif (CYLON_GLOO)

add_library(cylon SHARED
        ${UCX_CYLON_FILES}
        ${UCC_CYLON_FILES}
        ${CYLON_GLOO_FILES}
        ${UCX_REDIS_CYLON_FILES}
        arrow/arrow_all_to_all.cpp
        arrow/arrow_all_to_all.hpp
        arrow/arrow_buffer.cpp
        arrow/arrow_buffer.hpp
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
        compute/aggregate_kernels.cpp
        compute/aggregate_kernels.hpp
        compute/aggregate_utils.hpp
        compute/aggregates.cpp
        compute/aggregates.hpp
        compute/scalar_aggregate.cpp
        ctx/arrow_memory_pool_utils.cpp
        ctx/arrow_memory_pool_utils.hpp
        ctx/cylon_context.cpp
        ctx/cylon_context.hpp
        ctx/memory_pool.hpp
        data_types.cpp
        data_types.hpp
        groupby/groupby.cpp
        groupby/groupby.hpp
        groupby/hash_groupby.cpp
        groupby/hash_groupby.hpp
        groupby/pipeline_groupby.cpp
        groupby/pipeline_groupby.hpp
        indexing/index.cpp
        indexing/index.hpp
        indexing/index_utils.cpp
        indexing/index_utils.hpp
        indexing/indexer.cpp
        indexing/indexer.hpp
        indexing/slice.cpp
        io/arrow_io.cpp
        io/arrow_io.hpp
        io/csv_read_config.cpp
        io/csv_read_config.hpp
        io/csv_read_config_holder.hpp
        io/csv_write_config.cpp
        io/csv_write_config.hpp
        io/parquet_config.hpp
        io/parquet_config.cpp
        join/hash_join.cpp
        join/hash_join.hpp
        join/join.cpp
        join/join.hpp
        join/join_config.hpp
        join/join_utils.cpp
        join/join_utils.hpp
        join/sort_join.cpp
        join/sort_join.hpp
        mapreduce/mapreduce.hpp
        mapreduce/mapreduce.cpp
        net/channel.hpp
        net/comm_operations.hpp
        net/comm_type.hpp
        net/communicator.cpp
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
        net/cylon_request.cpp
        net/cylon_request.hpp
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
        serialize/table_serialize.cpp
        serialize/table_serialize.hpp
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
        util/PackString.hpp
        util/PackString.cpp
        scalar.cpp
        scalar.hpp
        net/ops/base_ops.hpp
        net/ops/base_ops.cpp
        net/utils.cpp
        net/utils.hpp)

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
target_link_libraries(cylon ${ARROW_DATASET_LIB})
target_link_libraries(cylon Threads::Threads)
target_link_libraries(cylon ${PARQUET_LIB})


if (CYLON_UCX)
    message("Cylon UCX Enabled")
    target_link_libraries(cylon ${UCX_LIBRARIES})
    if (CYLON_UCC)
        message("Cylon UCC Enabled")
        target_link_libraries(cylon ucc)
    endif ()
    if(CYLON_USE_REDIS)
        message("Cylon Redis Enabled")
        add_definitions(-DBUILD_CYLON_REDIS)
        # <------------ add hiredis dependency --------------->
        find_path(HIREDIS_HEADER hiredis HINTS ${REDIS_INSTALL_PREFIX}/include)
        target_include_directories(cylon PUBLIC ${HIREDIS_HEADER})

        #find_library(HIREDIS_LIB hiredis HINTS ${REDIS_INSTALL_PREFIX}/lib)
        #target_link_libraries(cylon ${HIREDIS_LIB})

        # <------------ add redis-plus-plus dependency -------------->
        # NOTE: this should be *sw* NOT *redis++*
        find_path(REDIS_PLUS_PLUS_HEADER sw HINTS ${REDIS_INSTALL_PREFIX}/include)
        target_include_directories(cylon PUBLIC ${REDIS_PLUS_PLUS_HEADER})

        #find_library(REDIS_PLUS_PLUS_LIB redis++ HINTS ${REDIS_INSTALL_PREFIX}/lib)
        #target_link_libraries(cylon ${REDIS_PLUS_PLUS_LIB})
        target_link_libraries(cylon ${REDIS_LIBRARIES})
    endif()
endif ()

if (CYLON_GLOO)
    target_link_libraries(cylon ${GLOO_LIBRARIES})
endif ()

if (PYCYLON_BUILD)
    target_link_libraries(cylon ${Python3_LIBRARIES})
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

install(TARGETS cylon LIBRARY DESTINATION lib)
install(TARGETS cylon RUNTIME DESTINATION bin)