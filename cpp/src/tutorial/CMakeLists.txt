
#build a binary
macro(tx_add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp)
    target_link_libraries(${EXENAME} ${MPI_LIBRARIES})
    target_link_libraries(${EXENAME} cylon)
    target_link_libraries(${EXENAME} ${ARROW_LIB})
    target_link_libraries(${EXENAME} ${GLOG_LIBRARIES})
endmacro(tx_add_exe)

include_directories(../cylon)

tx_add_exe(demo_join)

file(GLOB PYTORCH_DATA_1 "${CMAKE_CURRENT_SOURCE_DIR}/data/user_device_tm*.csv")
file(GLOB PYTORCH_DATA_2 "${CMAKE_CURRENT_SOURCE_DIR}/data/user_usage_tm*.csv")
file(COPY ${PYTORCH_DATA_1} DESTINATION /tmp/)
file(COPY ${PYTORCH_DATA_2} DESTINATION /tmp/)