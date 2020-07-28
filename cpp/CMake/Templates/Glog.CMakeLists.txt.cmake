cmake_minimum_required(VERSION 3.10)

include(ExternalProject)

ExternalProject_Add(glog
        GIT_REPOSITORY    "https://github.com/google/glog.git"
        GIT_TAG           v0.3.5
        SOURCE_DIR        "${GLOG_ROOT}/glog"
        BINARY_DIR        "${GLOG_ROOT}/build"
        INSTALL_DIR       "${GLOG_ROOT}/install"
        CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=${GLOG_ROOT}/install -DWITH_GFLAGS=OFF)