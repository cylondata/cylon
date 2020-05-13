#build a binary
macro(tx_add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp)
endmacro(tx_add_exe)

include_directories(../twisterx)

tx_add_exe(arrow_twisterx)
tx_add_exe(arrow_test)
tx_add_exe(join_test)
tx_add_exe(table_api_test)
tx_add_exe(table_api_test_hash)
tx_add_exe(hash_partition)
tx_add_exe(test_util)
tx_add_exe(test_net)
tx_add_exe(test_cy_alltoallwrap)

target_link_libraries(arrow_twisterx ${MPI_LIBRARIES})
target_link_libraries(arrow_twisterx twisterx)
target_link_libraries(arrow_twisterx ${ARROW_LIB})

target_link_libraries(arrow_test ${MPI_LIBRARIES})
target_link_libraries(arrow_test twisterx)
target_link_libraries(arrow_test ${ARROW_LIB})
target_link_libraries(arrow_test glog::glog)

target_link_libraries(join_test ${MPI_LIBRARIES})
target_link_libraries(join_test twisterx)
target_link_libraries(join_test ${ARROW_LIB})
target_link_libraries(join_test glog::glog)

target_link_libraries(table_api_test ${MPI_LIBRARIES})
target_link_libraries(table_api_test twisterx)
target_link_libraries(table_api_test ${ARROW_LIB})
target_link_libraries(table_api_test glog::glog)

target_link_libraries(table_api_test_hash.cpp ${MPI_LIBRARIES} twisterx ${ARROW_LIB} glog::glog)

target_link_libraries(hash_partition ${MPI_LIBRARIES})
target_link_libraries(hash_partition twisterx)
target_link_libraries(hash_partition ${ARROW_LIB})
target_link_libraries(hash_partition glog::glog)

target_link_libraries(test_util twisterx)
target_link_libraries(test_util ${ARROW_LIB})
target_link_libraries(test_util glog::glog)

target_link_libraries(test_net twisterx)
target_link_libraries(test_net ${ARROW_LIB})
target_link_libraries(test_net glog::glog)

target_link_libraries(test_cy_alltoallwrap twisterx)
target_link_libraries(test_cy_alltoallwrap ${ARROW_LIB})
target_link_libraries(test_cy_alltoallwrap glog::glog)