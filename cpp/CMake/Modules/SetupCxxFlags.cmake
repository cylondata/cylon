# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Inspired from SetupCxxFlags.cmake from Apache Arrow

# Check if the target architecture and compiler supports some special
# instruction sets that would boost performance.
include(CheckCXXCompilerFlag)
# Get cpu architecture

message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")

if(NOT DEFINED CYLON_CPU_FLAG)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
        set(CYLON_CPU_FLAG "armv8")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7")
        set(CYLON_CPU_FLAG "armv7")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc")
        set(CYLON_CPU_FLAG "ppc")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "s390x")
        set(CYLON_CPU_FLAG "s390x")
    else()
        set(CYLON_CPU_FLAG "x86")
    endif()
endif()

# Check architecture specific compiler flags
if(CYLON_CPU_FLAG STREQUAL "x86")
    # x86/amd64 compiler flags, msvc/gcc/clang
    if(MSVC)
        set(CYLON_SSE4_2_FLAG "")
        set(CYLON_AVX2_FLAG "/arch:AVX2")
        set(CYLON_AVX512_FLAG "/arch:AVX512")
        set(CXX_SUPPORTS_SSE4_2 TRUE)
    else()
        set(CYLON_SSE4_2_FLAG "-msse4.2")
        set(CYLON_AVX2_FLAG "-march=haswell")
        # skylake-avx512 consists of AVX512F,AVX512BW,AVX512VL,AVX512CD,AVX512DQ
        set(CYLON_AVX512_FLAG "-march=skylake-avx512 -mbmi2")
        # Append the avx2/avx512 subset option also, fix issue ARROW-9877 for homebrew-cpp
        set(CYLON_AVX2_FLAG "${CYLON_AVX2_FLAG} -mavx2")
        set(CYLON_AVX512_FLAG
                "${CYLON_AVX512_FLAG} -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw")
        check_cxx_compiler_flag(${CYLON_SSE4_2_FLAG} CXX_SUPPORTS_SSE4_2)
    endif()
    check_cxx_compiler_flag(${CYLON_AVX2_FLAG} CXX_SUPPORTS_AVX2)
    if(MINGW)
        # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65782
        message(STATUS "Disable AVX512 support on MINGW for now")
    else()
        check_cxx_compiler_flag(${CYLON_AVX512_FLAG} CXX_SUPPORTS_AVX512)
    endif()
    # Runtime SIMD level it can get from compiler
    if(CXX_SUPPORTS_SSE4_2
            AND CYLON_RUNTIME_SIMD_LEVEL MATCHES "^(SSE4_2|AVX2|AVX512|MAX)$")
        add_definitions(-DCYLON_HAVE_RUNTIME_SSE4_2)
    endif()
    if(CXX_SUPPORTS_AVX2 AND CYLON_RUNTIME_SIMD_LEVEL MATCHES "^(AVX2|AVX512|MAX)$")
        add_definitions(-DCYLON_HAVE_RUNTIME_AVX2 -DCYLON_HAVE_RUNTIME_BMI2)
    endif()
    if(CXX_SUPPORTS_AVX512 AND CYLON_RUNTIME_SIMD_LEVEL MATCHES "^(AVX512|MAX)$")
        add_definitions(-DCYLON_HAVE_RUNTIME_AVX512 -DCYLON_HAVE_RUNTIME_BMI2)
    endif()
elseif(CYLON_CPU_FLAG STREQUAL "ppc")
    # power compiler flags, gcc/clang only
    set(CYLON_ALTIVEC_FLAG "-maltivec")
    check_cxx_compiler_flag(${CYLON_ALTIVEC_FLAG} CXX_SUPPORTS_ALTIVEC)
elseif(CYLON_CPU_FLAG STREQUAL "armv8")
    # Arm64 compiler flags, gcc/clang only
    set(CYLON_ARMV8_ARCH_FLAG "-march=${CYLON_ARMV8_ARCH}")
    check_cxx_compiler_flag(${CYLON_ARMV8_ARCH_FLAG} CXX_SUPPORTS_ARMV8_ARCH)
endif()

# Only enable additional instruction sets if they are supported
if(CYLON_CPU_FLAG STREQUAL "x86")
    message("Required CYLON_SIMD_LEVEL: ${CYLON_SIMD_LEVEL}")
    if(CYLON_SIMD_LEVEL STREQUAL "AVX512")
        if(NOT CXX_SUPPORTS_AVX512)
            message(FATAL_ERROR "AVX512 required but compiler doesn't support it.")
        endif()
        set(CXX_COMMON_FLAGS "${CXX_COMMON_FLAGS} ${CYLON_AVX512_FLAG}")
        add_definitions(-DCYLON_HAVE_AVX512 -DCYLON_HAVE_AVX2 -DCYLON_HAVE_BMI2
                -DCYLON_HAVE_SSE4_2)
    elseif(CYLON_SIMD_LEVEL STREQUAL "AVX2")
        if(NOT CXX_SUPPORTS_AVX2)
            message(FATAL_ERROR "AVX2 required but compiler doesn't support it.")
        endif()
        set(CXX_COMMON_FLAGS "${CXX_COMMON_FLAGS} ${CYLON_AVX2_FLAG}")
        add_definitions(-DCYLON_HAVE_AVX2 -DCYLON_HAVE_BMI2 -DCYLON_HAVE_SSE4_2)
    elseif(CYLON_SIMD_LEVEL STREQUAL "SSE4_2")
        if(NOT CXX_SUPPORTS_SSE4_2)
            message(FATAL_ERROR "SSE4.2 required but compiler doesn't support it.")
        endif()
        set(CXX_COMMON_FLAGS "${CXX_COMMON_FLAGS} ${CYLON_SSE4_2_FLAG}")
        add_definitions(-DCYLON_HAVE_SSE4_2)
    endif()
endif()

if(CYLON_CPU_FLAG STREQUAL "ppc")
    if(CXX_SUPPORTS_ALTIVEC AND CYLON_ALTIVEC)
        set(CXX_COMMON_FLAGS "${CXX_COMMON_FLAGS} ${CYLON_ALTIVEC_FLAG}")
    endif()
endif()

if(CYLON_CPU_FLAG STREQUAL "armv8")
    if(NOT CXX_SUPPORTS_ARMV8_ARCH)
        message(FATAL_ERROR "Unsupported arch flag: ${CYLON_ARMV8_ARCH_FLAG}.")
    endif()
    if(CYLON_ARMV8_ARCH_FLAG MATCHES "native")
        message(FATAL_ERROR "native arch not allowed, please specify arch explicitly.")
    endif()
    set(CXX_COMMON_FLAGS "${CXX_COMMON_FLAGS} ${CYLON_ARMV8_ARCH_FLAG}")

    add_definitions(-DCYLON_HAVE_NEON)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
            AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.4")
        message(WARNING "Disable Armv8 CRC and Crypto as compiler doesn't support them well.")
    else()
        if(CYLON_ARMV8_ARCH_FLAG MATCHES "\\+crypto")
            add_definitions(-DCYLON_HAVE_ARMV8_CRYPTO)
        endif()
        # armv8.1+ implies crc support
        if(CYLON_ARMV8_ARCH_FLAG MATCHES "armv8\\.[1-9]|\\+crc")
            add_definitions(-DCYLON_HAVE_ARMV8_CRC)
        endif()
    endif()
endif()

