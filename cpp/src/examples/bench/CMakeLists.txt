#build a binary
macro(tx_add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp)
    target_link_libraries(${EXENAME} ${MPI_LIBRARIES})
    target_link_libraries(${EXENAME} cylon)
    target_link_libraries(${EXENAME} ${ARROW_LIB})
    target_link_libraries(${EXENAME} ${GLOG_LIBRARIES})
    install (TARGETS ${EXENAME} RUNTIME DESTINATION examples)
endmacro(tx_add_exe)

include_directories(../cylon)

tx_add_exe(table_join_dist_test)
tx_add_exe(table_union_dist_test)
tx_add_exe(table_agg_dist_test)
tx_add_exe(table_grp_dist_test)