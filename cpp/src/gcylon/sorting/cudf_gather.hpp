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

#ifndef GCYLON_CUDF_GATHER_HPP
#define GCYLON_CUDF_GATHER_HPP

#include <cylon/ctx/cylon_context.hpp>
#include <cylon/status.hpp>
#include <cylon/net/buffer.hpp>
#include <gcylon/sorting/serialize.hpp>

namespace gcylon {

class TableGatherer {
public:
    TableGatherer(std::shared_ptr<cylon::CylonContext> ctx,
                  const int gather_root,
                  std::shared_ptr<cylon::Allocator> allocator);

    cylon::Status Gather(cudf::table_view &tv,
                         bool gather_from_root,
                         std::vector<std::unique_ptr<cudf::table>> &gathered_tables);

    bool AmIRoot();

    std::vector<int32_t> totalBufferSizes(int32_t * all_buffer_sizes, int buffer_size_pw);
    std::vector<int32_t> receiveCounts(int32_t *all_buffer_sizes, int receiveNo, int buffer_size_pw);
    std::vector<int32_t> displacementsPerBuffer(int32_t *all_buffer_sizes, int receiveNo, int buffer_size_pw);
    std::vector<std::vector<int32_t>> bufferSizesPerTable(int32_t *all_buffer_sizes, int buffer_size_pw);

private:
    std::shared_ptr<cylon::CylonContext> ctx_;
    const int root_;
    std::shared_ptr<cylon::Allocator> allocator_;
};


} // end of namespace gcylon

#endif //GCYLON_CUDF_GATHER_HPP
