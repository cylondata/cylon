##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

# build glog from github tag
set(GLOG_ROOT ${CMAKE_BINARY_DIR}/glog)
set(GLOG_INSTALL ${CMAKE_BINARY_DIR}/glog/install)

#if (UNIX)
#    set(GLOG_EXTRA_COMPILER_FLAGS "-fPIC")
#endif()
IF(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
ENDIF()

set(GLOG_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${GLOG_EXTRA_COMPILER_FLAGS})
set(GLOG_C_FLAGS ${CMAKE_C_FLAGS} ${GLOG_EXTRA_COMPILER_FLAGS})

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
        COMMAND ${CMAKE_COMMAND} --build ..
        RESULT_VARIABLE GLOG_BUILD
        WORKING_DIRECTORY ${GLOG_ROOT}/build)

if (GLOG_BUILD)
    message(FATAL_ERROR "Building Glog failed: " ${GLOG_BUILD})
endif (GLOG_BUILD)

message(STATUS "Glog installed here: " ${GLOG_ROOT}/install)
if (EXISTS "${GLOG_ROOT}/install/lib")
    set(GLOG_LIBRARY_DIR "${GLOG_ROOT}/install/lib")
elseif (EXISTS "${GLOG_ROOT}/install/lib64")
    set(GLOG_LIBRARY_DIR "${GLOG_ROOT}/install/lib64")
else ()
    message(ERROR "Unable to find glog lib directory in ${GLOG_ROOT}/install")
endif ()
set(GLOG_INCLUDE_DIR "${GLOG_ROOT}/install/include")

set(GLOG_FOUND TRUE)

IF (WIN32)
    set(GLOG_LIBRARIES ${GLOG_INSTALL}/lib/glog.lib)
ELSE ()
    set(GLOG_LIBRARIES ${GLOG_LIBRARY_DIR}/libglog.a)
ENDIF ()

message(STATUS "Glog libs dir: " ${GLOG_LIBRARY_DIR})
message(STATUS "Glog include dir: " ${GLOG_INCLUDE_DIR})
message(STATUS "Glog libs: " ${GLOG_LIBRARIES})