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

#include <numeric>
#include <cylon/util/macros.hpp>
#include <gcylon/gtable_api.hpp>
#include <gcylon/net/cudf_net_ops.hpp>

namespace gcylon {

std::vector<int32_t> DivideRowsEvenly(int32_t all_rows, int32_t nworkers) {
  int32_t rows_per_worker = all_rows / nworkers;
  int32_t workers_with_extra_row = all_rows % nworkers;
  std::vector<int32_t> even_rows(nworkers, rows_per_worker);
  std::fill(even_rows.begin(), even_rows.begin() + workers_with_extra_row, rows_per_worker + 1);
  return even_rows;
}

int32_t RowGoesToWorker(int32_t all_rows, int32_t nworkers, int32_t row_no) {
  return nworkers * row_no / all_rows;
}

std::vector<int32_t> RowIndicesToAll(int32_t my_rank,
                                     int32_t nworkers,
                                     const std::vector<int32_t> &current_row_counts) {

  int32_t sum_of_rows = std::accumulate(current_row_counts.begin(), current_row_counts.end(), 0);
  auto rows_evenly_dist = DivideRowsEvenly(sum_of_rows, nworkers);

  int32_t my_row_start = std::accumulate(current_row_counts.begin(), current_row_counts.begin() + my_rank, 0);

  int32_t first_target_rank = RowGoesToWorker(sum_of_rows, nworkers, my_row_start);
  int32_t first_target_start_row = std::accumulate(rows_evenly_dist.begin(),
                                                   rows_evenly_dist.begin() + first_target_rank,
                                                   0);
  int32_t first_target_rows = rows_evenly_dist[first_target_rank] - (my_row_start - first_target_start_row);
  if (first_target_rows > current_row_counts[my_rank]) {
    first_target_rows = current_row_counts[my_rank];
  }

  std::vector<int32_t> to_all(rows_evenly_dist.size() + 1, 0);
  to_all[first_target_rank + 1] = first_target_rows;
  int remaining_rows = current_row_counts[my_rank] - first_target_rows;
  int next_target = first_target_rank + 1;

  while(remaining_rows > 0) {
    int32_t current_target_rows = remaining_rows > rows_evenly_dist[next_target] ? rows_evenly_dist[next_target] : remaining_rows;
    to_all[next_target + 1] = to_all[next_target] + current_target_rows;
    remaining_rows -= rows_evenly_dist[next_target];
    ++next_target;
  }

  ++next_target;
  // fill the remaining target values with the last row index, if any
  if (next_target < to_all.size()) {
    std::fill(to_all.begin() + next_target, to_all.end(), current_row_counts[my_rank]);
  }

  return to_all;
}

cylon::Status Repartition(const cudf::table_view &input_tv,
                          const std::shared_ptr<cylon::CylonContext> &ctx,
                          std::unique_ptr<cudf::table> &table_out){

  std::vector<int32_t> current_row_counts;
  RETURN_CYLON_STATUS_IF_FAILED(
    RowCountsAllTables(input_tv.num_rows(), ctx, current_row_counts));

  auto rows_to_all = RowIndicesToAll(ctx->GetRank(), ctx->GetWorldSize(), current_row_counts);

  RETURN_CYLON_STATUS_IF_FAILED(
    gcylon::net::AllToAll(input_tv, rows_to_all, ctx, table_out));

  return cylon::Status::OK();
}

}// end of namespace gcylon
