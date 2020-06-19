#build a binary
macro(tx_add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp)
endmacro(tx_add_exe)

include_directories(../../twisterx)

tx_add_exe(test_net)
tx_add_exe(test_cy_alltoallwrap)
tx_add_exe(distributed_context)
tx_add_exe(distributed_rl)

target_link_libraries(test_net twisterx_python)
target_link_libraries(test_net ${ARROW_LIB})
target_link_libraries(test_net glog::glog)

target_link_libraries(test_cy_alltoallwrap twisterx_python)
target_link_libraries(test_cy_alltoallwrap ${ARROW_LIB})
target_link_libraries(test_cy_alltoallwrap glog::glog)

target_link_libraries(distributed_context twisterx_python)
target_link_libraries(distributed_context ${ARROW_LIB})
target_link_libraries(distributed_context glog::glog)

target_link_libraries(distributed_rl twisterx_python)
target_link_libraries(distributed_rl ${ARROW_LIB})
target_link_libraries(distributed_rl glog::glog)

