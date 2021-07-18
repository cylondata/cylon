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
file(GLOB DUPLICATE_DATA_1 "${CMAKE_CURRENT_SOURCE_DIR}/data/duplicate_data_*.csv")
file(COPY ${PYTORCH_DATA_1} DESTINATION /tmp/)
file(COPY ${PYTORCH_DATA_2} DESTINATION /tmp/)
file(COPY ${DUPLICATE_DATA_1} DESTINATION /tmp/)
