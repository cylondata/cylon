#build a binary
macro(tx_add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp)
    target_link_libraries(${EXENAME} ${MPI_LIBRARIES})
    target_link_libraries(${EXENAME} cylon)
    target_link_libraries(${EXENAME} ${ARROW_LIB})
    target_link_libraries(${EXENAME} ${GLOG_LIBRARIES})
    install(TARGETS ${EXENAME} RUNTIME DESTINATION examples)
endmacro(tx_add_exe)

include_directories(../cylon)

tx_add_exe(union_example)
tx_add_exe(select_example)
tx_add_exe(join_example)
tx_add_exe(project_example)
tx_add_exe(subtract_example)
tx_add_exe(intersect_example)
tx_add_exe(task_test)
tx_add_exe(join_inmem_datagen_example)
tx_add_exe(com_comm_test)
tx_add_exe(table_from_vectors_example)
tx_add_exe(join_op_example)
tx_add_exe(compute_example)


#macro(tx_add_test_exe EXENAME)
#    add_executable(${EXENAME} ${EXENAME}.cpp ../../test/test_utils.hpp)
#    target_link_libraries(${EXENAME} ${MPI_LIBRARIES})
#    target_link_libraries(${EXENAME} cylon)
#    target_link_libraries(${EXENAME} ${ARROW_LIB})
#    install (TARGETS ${EXENAME} RUNTIME DESTINATION examples)
#endmacro(tx_add_test_exe)
#tx_add_test_exe(join_test)
#tx_add_test_exe(set_op_test)


