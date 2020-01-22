#build a binary
macro(tx_add_exe EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cpp)
endmacro(tx_add_exe)

tx_add_exe(small_twisterx)

if(MPI_FOUND)
  include_directories(${MPI_INCLUDE_PATH})
  target_link_libraries(small_twisterx ${MPI_LIBRARIES})
endif()

