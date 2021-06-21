# Set paths
set(UCX_HOME ${CMAKE_BINARY_DIR}/ucx/install)
set(UCX_INSTALL ${CMAKE_BINARY_DIR}/ucx/install)
set(UCX_ROOT ${CMAKE_BINARY_DIR}/ucx)

# Set to create the configure file from the template
configure_file("${CMAKE_SOURCE_DIR}/CMake/Templates/UCX.CMakeLists.txt.cmake"
        "${UCX_ROOT}/CMakeLists.txt")

# Make directories
file(MAKE_DIRECTORY "${UCX_ROOT}/install")
file(MAKE_DIRECTORY "${UCX_ROOT}/build")

# Generate the actual CMake file from config
execute_process(
        COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE UCX_CONFIG
        WORKING_DIRECTORY ${UCX_ROOT})

# Check config generation
if (UCX_CONFIG)
    message(FATAL_ERROR "Configuring UCX failed: " ${UCX_CONFIG})
endif (UCX_CONFIG)

# Build UCX
execute_process(
        COMMAND ${CMAKE_COMMAND} --build ..
        RESULT_VARIABLE UCX_BUILD
        WORKING_DIRECTORY ${UCX_ROOT}/build)

# Check the status of the build
if(UCX_BUILD)
    message(FATAL_ERROR "Building UCX failed: " ${UCX_BUILD})
endif(UCX_BUILD)

# Message of result
message(STATUS "UCX installed here: " ${UCX_ROOT}/install)
# Set paths for using UCX
set(UCX_LIBRARY_DIR "${UCX_ROOT}/install/lib")
set(UCX_INCLUDE_DIR "${UCX_ROOT}/install/include")

# Set UCX found as true
set(UCX_FOUND TRUE)
# Specify the libraries of UCX
set(UCX_LIBRARIES
        ${UCX_INSTALL}/lib/ucx/libuct_cma.so
        ${UCX_INSTALL}/lib/ucx/libuct_ib.so
        ${UCX_INSTALL}/lib/libuct.so
        ${UCX_INSTALL}/lib/libucs.so
        ${UCX_INSTALL}/lib/libucm.so
        ${UCX_INSTALL}/lib/libucp.so)