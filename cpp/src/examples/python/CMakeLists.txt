#build a binary
macro(tx_add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp)
endmacro(tx_add_exe)

include_directories(../../cylon)

tx_add_exe(test_net)
tx_add_exe(test_cy_alltoallwrap)
tx_add_exe(distributed_context)
tx_add_exe(distributed_rl)

target_link_libraries(test_net cylon_python)
target_link_libraries(test_net ${ARROW_LIB})
target_link_libraries(test_net ${GLOG_LIBRARIES})

target_link_libraries(test_cy_alltoallwrap cylon_python)
target_link_libraries(test_cy_alltoallwrap ${ARROW_LIB})
target_link_libraries(test_cy_alltoallwrap ${GLOG_LIBRARIES})

target_link_libraries(distributed_context cylon_python)
target_link_libraries(distributed_context ${ARROW_LIB})
target_link_libraries(distributed_context ${GLOG_LIBRARIES})

target_link_libraries(distributed_rl cylon_python)
target_link_libraries(distributed_rl ${ARROW_LIB})
target_link_libraries(distributed_rl ${GLOG_LIBRARIES})

