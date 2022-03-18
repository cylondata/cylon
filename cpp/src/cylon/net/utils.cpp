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

#include "utils.hpp"


std::vector<int32_t>
cylon::net::ReshapeDispToPerTable(const std::vector<std::vector<int32_t>> &all_disps) {
  const size_t num_buf = all_disps.size();
  const size_t num_tables = all_disps[0].size(); // == world_size

  std::vector<int32_t> res(num_buf * num_tables, 0);
  for (size_t tid = 0; tid < num_tables; tid++) {
    for (size_t bid = 0; bid < num_buf; bid++) {
      res[tid * num_buf + bid] = all_disps[bid][tid];
    }
  }
  return res;
}

std::vector<int32_t> cylon::net::totalBufferSizes(const std::vector<int32_t> &all_buffer_sizes,
                                                  int num_buffers,
                                                  int world_size)  {
  std::vector<int32_t> total_buffer_sizes(num_buffers, 0);
  for (int w = 0; w < world_size; w++) {
    for (int i = 0; i < num_buffers; i++) {
      total_buffer_sizes[i] += all_buffer_sizes[w * num_buffers + i];
    }
  }
  return total_buffer_sizes;
}

std::vector<int32_t> cylon::net::receiveCounts(const std::vector<int32_t> &all_buffer_sizes,
                                               int receiveNo,
                                               int num_buffers,
                                               int world_size)  {
  std::vector<int32_t> receive_counts(world_size, 0);
  for (int i = 0; i < world_size; ++i) {
    receive_counts[i] = all_buffer_sizes[i * num_buffers + receiveNo];
  }
  return receive_counts;
}

std::vector<int32_t> cylon::net::displacementsPerBuffer(const std::vector<int32_t> &all_buffer_sizes,
                                                        int receiveNo,
                                                        int num_buffers,
                                                        int world_size)  {
  std::vector<int32_t> disp_array(world_size, 0);
  disp_array[0] = 0;
  for (int i = 0; i < world_size - 1; ++i) {
    disp_array[i + 1] = disp_array[i] + all_buffer_sizes[i * num_buffers + receiveNo];
  }
  return disp_array;
}
