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

# This code was taken from the Apache Arrow project
function(CYLON_INSTALL_ALL_HEADERS PATH)
    set(options)
    set(one_value_args)
    set(multi_value_args PATTERN)
    cmake_parse_arguments(ARG
            "${options}"
            "${one_value_args}"
            "${multi_value_args}"
            ${ARGN})
    if(NOT ARG_PATTERN)
        # The .hpp extension is used by some vendored libraries
        set(ARG_PATTERN "*.h" "*.hpp")
    endif()
    file(GLOB CURRENT_DIRECTORY_HEADERS ${ARG_PATTERN})

    set(PUBLIC_HEADERS)
    foreach(HEADER ${CURRENT_DIRECTORY_HEADERS})
        get_filename_component(HEADER_BASENAME ${HEADER} NAME)
        if(HEADER_BASENAME MATCHES "internal")
            continue()
        endif()
        list(APPEND PUBLIC_HEADERS ${HEADER})
    endforeach()
#    message(${PUBLIC_HEADERS})
    install(FILES ${PUBLIC_HEADERS} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PATH}")
endfunction()