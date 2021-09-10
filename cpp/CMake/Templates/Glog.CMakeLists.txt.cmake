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

cmake_minimum_required(VERSION 3.10)
project(GlogModule)
include(ExternalProject)

ExternalProject_Add(glog
        GIT_REPOSITORY "https://github.com/google/glog.git"
        GIT_TAG v0.5.0
        SOURCE_DIR "${GLOG_ROOT}/glog"
        BINARY_DIR "${GLOG_ROOT}/build"
        INSTALL_DIR "${GLOG_ROOT}/install"
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${GLOG_ROOT}/install -DWITH_GFLAGS=OFF -DWITH_UNWIND=OFF -DBUILD_SHARED_LIBS=OFF)