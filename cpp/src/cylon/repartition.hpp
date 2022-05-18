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

namespace cylon {

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
template<typename T>
std::vector<T> DivideRowsEvenly(size_t nworkers, const T total_rows) {
  T rows_per_worker = total_rows / nworkers;
  T workers_with_extra_row = total_rows % nworkers;
  std::vector<T> even_rows(nworkers, rows_per_worker);
  std::fill(even_rows.begin(), even_rows.begin() + workers_with_extra_row, rows_per_worker + 1);
  return even_rows;
}

template <typename T>
std::vector<T> DivideRowsEvenly(const std::vector<T> &row_counts) {
  auto nworkers = row_counts.size();
  auto all_rows = std::accumulate(row_counts.begin(), row_counts.end(), 0);
  return DivideRowsEvenly(nworkers, all_rows);
}

/**
 * determine the target worker of the first row
 * calculate the global order of the first row of that worker
 * @param target_row_counts
 * @param my_row_start
 * @return
 */
template<typename T>
std::pair<T, T> FirstTarget(const std::vector<T> &target_row_counts, T my_row_start) {
  // target worker of my first row
  T first_target_rank = -1;
  // global order of the first row of the first target worker
  T first_target_start_row = 0;
  for (size_t i = 0; i < target_row_counts.size(); ++i) {
    if (first_target_start_row + target_row_counts[i] > my_row_start) {
      first_target_rank =static_cast<T>( i);
      break;
    } else {
      first_target_start_row += target_row_counts[i];
    }
  }

  return std::make_pair(first_target_rank, first_target_start_row);
}

/**
 * calculates indicies for all-to-all operation for a worker
 * given the current row_counts and requested row counts
 * each worker calls this function to determine ranges of rows to send to each worker
 * In the returned vector:
 *   rows in the range of to_all[0] - to_all[1] shall be send to worker 0
 *   rows in the range of to_all[1] - to_all[2] shall be send to worker 2
 *   ...
 *   rows in the range of to_all[size-2] - to_all[size-1] shall be send to the last worker
 *
 * make sure:
 *   number_of_workers == current_row_counts.size() == target_row_counts.size()
 * make sure:
 *   sum_of(current_row_counts) == sum_of(target_row_counts)
 * we do not check those in the function
 * @param my_rank
 * @param current_row_counts
 * @param target_row_counts
 * @return
 */
template<typename T>
std::vector<T> RowIndicesToAll(int32_t my_rank,
                               const std::vector<T> &current_row_counts,
                               const std::vector<T> &target_row_counts) {

  // global order of my current first row
  T my_row_start = std::accumulate(current_row_counts.begin(), current_row_counts.begin() + my_rank, 0);

  auto first_target = FirstTarget(target_row_counts, my_row_start);
  auto first_target_rank = first_target.first;
  auto first_target_start_row = first_target.second;

  int32_t rows_to_first_target = target_row_counts[first_target_rank] - (my_row_start - first_target_start_row);
  if (rows_to_first_target > current_row_counts[my_rank]) {
    rows_to_first_target = current_row_counts[my_rank];
  }

  std::vector<T> to_all(target_row_counts.size() + 1, 0);
  to_all[first_target_rank + 1] = rows_to_first_target;
  auto remaining_rows = current_row_counts[my_rank] - rows_to_first_target;
  auto next_target = first_target_rank + 1;

  while(remaining_rows > 0) {
    auto current_target_rows = remaining_rows > target_row_counts[next_target] ? target_row_counts[next_target] : remaining_rows;
    to_all[next_target + 1] = to_all[next_target] + current_target_rows;
    remaining_rows -= target_row_counts[next_target];
    ++next_target;
  }

  ++next_target;
  // fill the remaining target values with the last row index, if any
  if ((size_t)next_target < to_all.size()) {
    std::fill(to_all.begin() + next_target, to_all.end(), current_row_counts[my_rank]);
  }

  return to_all;
}

}// end of namespace cylon
