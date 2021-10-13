/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cylon/util/macros.hpp>
#include <glog/logging.h>
#include <cylon/net/mpi/mpi_operations.hpp>
#include <gcylon/cudf_buffer.hpp>
#include <gcylon/net/cudf_net_ops.hpp>
#include <gcylon/sorting/deserialize.hpp>
#include <gcylon/net/cudf_serialize.hpp>

#include <cudf/table/table.hpp>


std::vector<std::vector<int32_t>> bufferSizesPerTable(const std::vector<int32_t> &all_buffer_sizes,
                                                      int number_of_buffers,
                                                      int numb_workers) {
  std::vector<std::vector<int32_t>> buffer_sizes_all_tables;
  for (int i = 0, k = 0; i < numb_workers; ++i) {
    std::vector<int32_t> single_table_buffer_sizes;
    for (int j = 0; j < number_of_buffers; ++j) {
      single_table_buffer_sizes.push_back(all_buffer_sizes[k++]);
    }
    buffer_sizes_all_tables.push_back(single_table_buffer_sizes);
  }
  return buffer_sizes_all_tables;
}


cylon::Status gcylon::net::Gather(const cudf::table_view &tv,
                                  const int gather_root,
                                  bool gather_from_root,
                                  std::shared_ptr<cylon::CylonContext> ctx,
                                  std::vector<std::unique_ptr<cudf::table>> &gathered_tables) {

  auto serializer = std::make_shared<CudfTableSerializer>(tv);
  auto allocator = std::make_shared<CudfAllocator>();
  std::vector<int32_t> all_buffer_sizes;
  std::vector<std::shared_ptr<cylon::Buffer>> receive_buffers;
  std::vector<std::vector<int32_t>> all_disps;

  RETURN_CYLON_STATUS_IF_FAILED(cylon::mpi::Gather(serializer,
                                                   gather_root,
                                                   gather_from_root,
                                                   allocator,
                                                   all_buffer_sizes,
                                                   receive_buffers,
                                                   all_disps,
                                                   ctx));

  if (cylon::mpi::AmIRoot(gather_root, ctx)) {
    std::vector<std::vector<int32_t>> buffer_sizes_per_table =
        bufferSizesPerTable(all_buffer_sizes, receive_buffers.size(), ctx->GetWorldSize());

    TableDeserializer deserializer(tv);
    deserializer.deserialize(receive_buffers, all_disps, buffer_sizes_per_table, gathered_tables);
  }

  return cylon::Status::OK();
}
