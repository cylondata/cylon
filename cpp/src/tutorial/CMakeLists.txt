
#build a binary
macro(tx_add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp)
    target_link_libraries(${EXENAME} ${MPI_LIBRARIES})
    target_link_libraries(${EXENAME} cylon)
    target_link_libraries(${EXENAME} ${ARROW_LIB})
    target_link_libraries(${EXENAME} ${GLOG_LIBRARIES})
endmacro(tx_add_exe)

include_directories(../cylon)

tx_add_exe(simple_join)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/csv1.csv  DESTINATION /tmp/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/csv2.csv  DESTINATION /tmp/)
