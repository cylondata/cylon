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



cdef extern from "../../../../cpp/src/cylon/data_types.hpp" namespace "cylon":
    cdef cppclass CLayout 'cylon::Layout':
        enum clayout 'layout':
            CFIXED_WIDTH 'cylon::Layout::FIXED_WIDTH'
            CVARIABLE_WIDTH 'cylon::Layout::VARIABLE_WIDTH'