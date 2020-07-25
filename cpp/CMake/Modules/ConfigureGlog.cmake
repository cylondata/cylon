# build glog from github tag
set(GLOG_ROOT ${CMAKE_BINARY_DIR}/glog)
set(GLOG_INSTALL ${CMAKE_BINARY_DIR}/glog/install)

#if (UNIX)
#    set(GLOG_EXTRA_COMPILER_FLAGS "-fPIC")
#endif()

set(GLOG_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${GLOG_EXTRA_COMPILER_FLAGS})
set(GLOG_C_FLAGS ${CMAKE_C_FLAGS} ${GLOG_EXTRA_COMPILER_FLAGS})

SET(WITH_GFLAGS OFF)

configure_file("${CMAKE_SOURCE_DIR}/CMake/Templates/Glog.CMakeLists.txt.cmake"
        "${GLOG_ROOT}/CMakeLists.txt")

file(MAKE_DIRECTORY "${GLOG_ROOT}/build")
file(MAKE_DIRECTORY "${GLOG_ROOT}/install")

execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE GLOG_CONFIG
        WORKING_DIRECTORY ${GLOG_ROOT})

if(GLOG_CONFIG)
    message(FATAL_ERROR "Configuring Glog failed: " ${GLOG_CONFIG})
endif(GLOG_CONFIG)

execute_process(
        COMMAND ${CMAKE_COMMAND} --build .. -- -j 2
        RESULT_VARIABLE GLOG_BUILD
        WORKING_DIRECTORY ${GLOG_ROOT}/build)

if(GLOG_BUILD)
    message(FATAL_ERROR "Building Glog failed: " ${GLOG_BUILD})
endif(GLOG_BUILD)

message(STATUS "Glog installed here: " ${GLOG_ROOT}/install)
set(GLOG_LIBRARY_DIR "${GLOG_ROOT}/install/lib")
set(GLOG_INCLUDE_DIR "${GLOG_ROOT}/install/include")

set(GLOG_FOUND TRUE)
set(GLOG_LIBRARIES ${GLOG_INSTALL}/lib/libglog.a)

