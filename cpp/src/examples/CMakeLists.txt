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

add_definitions(-DEXAMPLE_CMAKE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")

add_library(cylon_example_utils SHARED
        example_utils.cpp
        example_utils.hpp)

target_link_libraries(cylon_example_utils ${MPI_CXX_LIBRARIES})
target_link_libraries(cylon_example_utils ${GLOG_LIBRARIES})
target_link_libraries(cylon_example_utils ${ARROW_LIB})
target_link_libraries(cylon_example_utils cylon)

IF(WIN32)
    target_compile_options(cylon_example_utils PRIVATE -Wall)
ELSE()
    target_compile_options(cylon_example_utils PRIVATE -Werror -Wall -Wextra -Wno-unused-parameter)
ENDIF()

install(TARGETS cylon_example_utils RUNTIME DESTINATION cylon/examples)

#build a binary
macro(cylon_add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp)
    target_link_libraries(${EXENAME} ${MPI_CXX_LIBRARIES})
    target_link_libraries(${EXENAME} cylon)
    target_link_libraries(${EXENAME} ${ARROW_LIB})
    target_link_libraries(${EXENAME} ${GLOG_LIBRARIES})
    target_link_libraries(${EXENAME} cylon_example_utils)
    install(TARGETS ${EXENAME} RUNTIME DESTINATION cylon/examples)
endmacro(cylon_add_exe)

include_directories(../cylon)

cylon_add_exe(union_example)
cylon_add_exe(select_example)
cylon_add_exe(join_example)
cylon_add_exe(project_example)
cylon_add_exe(subtract_example)
cylon_add_exe(intersect_example)
cylon_add_exe(table_from_vectors_example)
cylon_add_exe(compute_example)
cylon_add_exe(groupby_pipeline_example)
cylon_add_exe(groupby_example)
cylon_add_exe(unique_example)
cylon_add_exe(indexing_example)
cylon_add_exe(sorting_example)
cylon_add_exe(multicolumn_sorting_example)
cylon_add_exe(multi_idx_join_example)
cylon_add_exe(unique_perf)

if (CYLON_PARQUET)
    cylon_add_exe(parquet_union_example)
    cylon_add_exe(parquet_join_example)
endif (CYLON_PARQUET)

if (CYLON_UCX)
    cylon_add_exe(ucx_join_example)
endif (CYLON_UCX)

macro(cylon_add_exe2 DIR EXENAME)
    add_executable(${EXENAME} ${DIR}/${EXENAME}.cpp)
    target_link_libraries(${EXENAME} ${MPI_LIBRARIES})
    target_link_libraries(${EXENAME} cylon)
    target_link_libraries(${EXENAME} ${ARROW_LIB})
    target_link_libraries(${EXENAME} ${GLOG_LIBRARIES})
    target_link_libraries(${EXENAME} cylon_example_utils)
    install(TARGETS ${EXENAME} RUNTIME DESTINATION cylon/examples)
endmacro(cylon_add_exe2)



