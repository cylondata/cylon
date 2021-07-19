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