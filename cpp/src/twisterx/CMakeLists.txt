add_library(twisterx SHARED net/all_to_all.cpp net/all_to_all.hpp net/channel.hpp net/mpi_channel.hpp net/mpi_channel.cpp arrow/arrow_all_to_all.cpp arrow/arrow_all_to_all.hpp join/join.hpp join/join.cpp util/arrow_utils.hpp util/arrow_utils.cpp arrow/arrow_kernels.cpp arrow/arrow_kernels.hpp util/copy_arrray.cpp join/join_utils.hpp join/join_utils.cpp arrow/arrow_join.cpp arrow/arrow_join.hpp util/sort_indices.cpp lib/Circle.h lib/Circle.cpp io/arrow_io.cpp io/arrow_io.hpp table_api.hpp table_api.cpp status.cpp code.cpp util/to_string.hpp table.cpp table.hpp util/uuid.h column.cpp column.hpp data_types.hpp arrow/arrow_types.hpp arrow/arrow_types.cpp arrow/arrow_partition_kernels.hpp arrow/arrow_partition_kernels.cpp util/murmur3.cpp util/murmur3.hpp arrow/arrow_partition_kernels.cpp arrow/arrow_partition_kernels.hpp join/join_config.h io/csv_read_config.h io/csv_read_config.cpp util/builtins.h util/builtins.cpp net/TxRequest.h net/TxRequest.cpp data/table_builder.h data/table_builder.cpp)

include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(twisterx ${MPI_LIBRARIES})
target_link_libraries(twisterx glog::glog)
target_link_libraries(twisterx ${ARROW_LIB})
