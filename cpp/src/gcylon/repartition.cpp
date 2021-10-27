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

/**
 * Divide the rows evenly among all workers
 * number of workers = row_counts.size()
 * If the rows can not be divided evenly among all workers,
 * first k workers get one extra row.
 * For example:
 *   row_counts{0, 14, 0, 0}
 *   Result: {4, 4, 3, 3}
 * @param row_counts
 * @return
 */
std::vector<int32_t> DivideRowsEvenly(const std::vector<int32_t> &row_counts) {
  auto nworkers = row_counts.size();
  auto all_rows = std::accumulate(row_counts.begin(), row_counts.end(), 0);
  int32_t rows_per_worker = all_rows / nworkers;
  int32_t workers_with_extra_row = all_rows % nworkers;
  std::vector<int32_t> even_rows(nworkers, rows_per_worker);
  std::fill(even_rows.begin(), even_rows.begin() + workers_with_extra_row, rows_per_worker + 1);
  return even_rows;
}

/**
 * determine the target worker of the first row
 * calculate the global order of the first row of that worker
 * @param target_row_counts
 * @param my_row_start
 * @return
 */
std::pair<int32_t, int32_t> FirstTarget(const std::vector<int32_t> &target_row_counts, int32_t my_row_start) {
  // target worker of my first row
  int32_t first_target_rank = -1;
  // global order of the first row of the first target worker
  int32_t first_target_start_row = 0;
  for (int i = 0; i < target_row_counts.size(); ++i) {
    if (first_target_start_row + target_row_counts[i] > my_row_start) {
      first_target_rank = i;
      break;
    } else {
      first_target_start_row += target_row_counts[i];
    }
  }

  return std::make_pair(first_target_rank, first_target_start_row);
}

/**
 * calculates indicies for all-to-all operation for a worker
 * given the current row_counts and requested rows counts
 * make sure: number_of_workers == current_row_counts.size()
 * make sure: sum_of_rows == sum_of(current_row_counts) == sum_of(target_row_counts)
 * we do not check this in the function
 * @param my_rank
 * @param sum_of_rows sum of rows in distributed tables
 * @param current_row_counts
 * @param target_row_counts
 * @return
 */
std::vector<int32_t> RowIndicesToAll(int32_t my_rank,
                                     const std::vector<int32_t> &current_row_counts,
                                     const std::vector<int32_t> &target_row_counts) {

  // global order of my current first row
  auto my_row_start = std::accumulate(current_row_counts.begin(), current_row_counts.begin() + my_rank, 0);

  auto first_target = FirstTarget(target_row_counts, my_row_start);
  int32_t first_target_rank = first_target.first;
  int32_t first_target_start_row = first_target.second;

  int32_t rows_to_first_target = target_row_counts[first_target_rank] - (my_row_start - first_target_start_row);
  if (rows_to_first_target > current_row_counts[my_rank]) {
    rows_to_first_target = current_row_counts[my_rank];
  }

  std::vector<int32_t> to_all(target_row_counts.size() + 1, 0);
  to_all[first_target_rank + 1] = rows_to_first_target;
  int remaining_rows = current_row_counts[my_rank] - rows_to_first_target;
  int next_target = first_target_rank + 1;

  while(remaining_rows > 0) {
    int32_t current_target_rows = remaining_rows > target_row_counts[next_target] ? target_row_counts[next_target] : remaining_rows;
    to_all[next_target + 1] = to_all[next_target] + current_target_rows;
    remaining_rows -= target_row_counts[next_target];
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
                          std::unique_ptr<cudf::table> &table_out,
                          const std::vector<int32_t> &rows_per_worker){

  std::vector<int32_t> current_row_counts;
  RETURN_CYLON_STATUS_IF_FAILED(
    RowCountsAllTables(input_tv.num_rows(), ctx, current_row_counts));

  std::vector<int32_t> rows_to_all;
  if (rows_per_worker.empty()) {
    auto evenly_dist_rows = DivideRowsEvenly(current_row_counts);
    rows_to_all = RowIndicesToAll(ctx->GetRank(), current_row_counts, evenly_dist_rows);
  } else {
    int32_t sum_of_current_rows = std::accumulate(current_row_counts.begin(), current_row_counts.end(), 0);
    int32_t sum_of_target_rows = std::accumulate(rows_per_worker.begin(), rows_per_worker.end(), 0);
    if (sum_of_current_rows != sum_of_target_rows) {
      return cylon::Status(cylon::Code::ValueError,
                           "Sum of target partitions does not match the sum of current partitions.");
    }
    rows_to_all = RowIndicesToAll(ctx->GetRank(), current_row_counts, rows_per_worker);
  }

  RETURN_CYLON_STATUS_IF_FAILED(
    gcylon::net::AllToAll(input_tv, rows_to_all, ctx, table_out));

  return cylon::Status::OK();
}

}// end of namespace gcylon
