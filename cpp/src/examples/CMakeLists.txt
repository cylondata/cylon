#build a binary
macro(tx_add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp)
    target_link_libraries(${EXENAME} ${MPI_LIBRARIES})
    target_link_libraries(${EXENAME} cylon)
    target_link_libraries(${EXENAME} ${ARROW_LIB})
    install (TARGETS ${EXENAME} RUNTIME DESTINATION examples)
endmacro(tx_add_exe)

include_directories(../cylon)

tx_add_exe(union_example)
tx_add_exe(select_example)
tx_add_exe(join_example)
tx_add_exe(project_example)
tx_add_exe(subtract_example)
tx_add_exe(intersect_example)
tx_add_exe(mpi_test)

