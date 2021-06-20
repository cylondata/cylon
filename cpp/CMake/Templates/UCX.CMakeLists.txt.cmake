cmake_minimum_required(VERSION 3.10)

include(ExternalProject)

ExternalProject_Add(ucx
        GIT_REPOSITORY    "https://github.com/openucx/ucx.git"
        GIT_TAG           v1.9.0
        SOURCE_DIR        "${UCX_ROOT}/ucx"
        BUILD_IN_SOURCE   1
        INSTALL_DIR       "${UCX_ROOT}/install"
        CONFIGURE_COMMAND ${UCX_ROOT}/ucx/autogen.sh COMMAND ${UCX_ROOT}/ucx/contrib/configure-release --prefix=${UCX_ROOT}/install
        BUILD_COMMAND     make -j8
        INSTALL_COMMAND   make install)