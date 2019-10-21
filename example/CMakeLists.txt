#build a binary
macro(tx_add_exe EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cpp)
endmacro(tx_add_exe)


tx_add_exe(small_twisterx)
