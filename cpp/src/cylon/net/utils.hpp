/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#ifndef CYLON_CPP_SRC_CYLON_NET_UTILS_HPP_
#define CYLON_CPP_SRC_CYLON_NET_UTILS_HPP_

#include <vector>
#include <cstdint>

namespace cylon {
namespace net {

/*
    |t_0, ..., t_m-1|...|t_0, ..., t_m-1|
     <--- buf_0 --->     <--- buf_n --->
                  to
    |b_0, ..., b_n-1|...|b_0, ..., b_n-1|
     <--- tbl_0 --->     <--- tbl_m --->
 */
std::vector<int32_t> ReshapeDispToPerTable(const std::vector<std::vector<int32_t>> &all_disps);

std::vector<int32_t> totalBufferSizes(const std::vector<int32_t> &all_buffer_sizes,
                                      int num_buffers,
                                      int world_size);

std::vector<int32_t> receiveCounts(const std::vector<int32_t> &all_buffer_sizes,
                                   int receiveNo,
                                   int num_buffers,
                                   int world_size);

std::vector<int32_t> displacementsPerBuffer(const std::vector<int32_t> &all_buffer_sizes,
                                            int receiveNo,
                                            int num_buffers,
                                            int world_size);

}
}


#endif //CYLON_CPP_SRC_CYLON_NET_UTILS_HPP_
