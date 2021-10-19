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
#include <cylon/net/mpi/mpi_operations.hpp>
#include <gcylon/gtable_api.hpp>
#include <gcylon/net/cudf_net_ops.hpp>
#include <gcylon/sorting/sorting.hpp>

#include <cudf/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/sorting.hpp>
#include <cudf/merge.hpp>
#include <cudf/copying.hpp>
#include <cudf/search.hpp>
#include <cudf/concatenate.hpp>
#include <rmm/device_buffer.hpp>
#include <cuda.h>
#include <algorithm>


bool gcylon::MergeOrSort(int num_columns, int num_tables) {
  if (num_columns == 1) {
    return num_tables <= 5;
  }

  if (num_columns >= 2 && num_columns < 4) {
    return num_tables <= 100;
  }

  if (num_columns >= 4 && num_columns < 8) {
    return num_tables <= 20;
  }

  if (num_columns >= 8) {
    return num_tables <= 10;
  }

  return true;
}

/**
 * create a vector with monotonically increasing numbers
 * @param count
 * @return
 */
std::vector<cudf::size_type> monotonicVector(int count, int start = 0, int step = 1) {
  std::vector<cudf::size_type> vec(count);
  std::generate(vec.begin(), vec.end(), [n = start - step, step]()mutable {return n += step;});
  return vec;
}

/**
 * when sorting columns, we ned to specify orders for columns that are not sort columns
 * if set the order as the majority order for those columns
 * this should not be significant mostly, only in some very unlikely cases
 * @param column_orders
 * @return
 */
cudf::order SortOrder(const std::vector<cudf::order> &column_orders) {
  int asc_count = std::count(column_orders.begin(), column_orders.end(), cudf::order::ASCENDING);
  return asc_count >= column_orders.size() / 2.0 ? cudf::order::ASCENDING : cudf::order::DESCENDING;
}

/**
 * merge or sort tables
 * all tables in tbl_all will be deleted during this process to reclaim memory
 * when merging first column_orders.size() columns are used
 * when sorting all columns from 0 to num_columns are used
 * @param tv_all
 * @param column_orders
 * @param null_ordering
 * @param merged_table
 * @return
 */
cylon::Status MergeOrSortTables(std::vector<std::unique_ptr<cudf::table>> &tbl_all,
                                const std::vector<cudf::order> &column_orders,
                                cudf::null_order null_ordering,
                                std::unique_ptr<cudf::table> &merged_table) {

  if (tbl_all.size() == 0) {
    return cylon::Status(cylon::Code::ExecutionError, "nothing to merge");
  }

  int32_t num_columns = tbl_all[0]->num_columns();

  std::vector<cudf::table_view> tv_all;
  tv_all.reserve(tbl_all.size());
  std::transform(tbl_all.begin(), tbl_all.end(), std::back_inserter(tv_all),
                 [](const std::unique_ptr<cudf::table>& t){ return t->view();});

  if (gcylon::MergeOrSort(num_columns, tv_all.size())) {
    auto key_cols = monotonicVector(column_orders.size());
    std::vector<cudf::null_order> column_null_orders(column_orders.size(), null_ordering);
    merged_table = cudf::merge(tv_all, key_cols, column_orders, column_null_orders);
    // delete all tables to reclaim memory
    // no need to for the tables from this point on
    std::for_each(tbl_all.begin(), tbl_all.end(), [](std::unique_ptr<cudf::table>& tbl){ tbl.reset();});

    if (merged_table == nullptr) {
      return cylon::Status(cylon::Code::ExecutionError, "merge failed");
    }
  } else {
    auto single_tbl = cudf::concatenate(tv_all);
    // delete all tables to reclaim memory
    // no need to for the tables from this point on
    std::for_each(tbl_all.begin(), tbl_all.end(), [](std::unique_ptr<cudf::table>& tbl){ tbl.reset();});

    std::vector<cudf::order> column_orders2(column_orders);
    column_orders2.insert(column_orders2.end(), num_columns - column_orders2.size(), SortOrder(column_orders));
    std::vector<cudf::null_order> column_null_orders(num_columns, null_ordering);
    merged_table = cudf::sort(single_tbl->view(), column_orders2, column_null_orders);
    if (merged_table == nullptr) {
      return cylon::Status(cylon::Code::ExecutionError, "merge failed");
    }
  }

  return cylon::Status::OK();
}

cylon::Status DetermineSplitPoints(std::vector<std::unique_ptr<cudf::table>> &gathered_tables,
                                   const std::vector<cudf::order> &column_orders,
                                   cudf::null_order null_ordering,
                                   const std::shared_ptr<cylon::CylonContext> &ctx,
                                   std::unique_ptr<cudf::table> &split_points_table) {

  std::unique_ptr<cudf::table> sorted_table;
  RETURN_CYLON_STATUS_IF_FAILED(
    MergeOrSortTables(gathered_tables, column_orders, null_ordering, sorted_table));

  int num_split_points = ctx->GetWorldSize() - 1;
  auto sorted_tv = sorted_table->view();
  return gcylon::SampleTableUniform(sorted_tv, num_split_points, split_points_table);
}

cylon::Status gcylon::GetSplitPoints(std::unique_ptr<cudf::table> sample_tbl,
                                     int splitter,
                                     const std::vector<cudf::order> &column_orders,
                                     cudf::null_order null_ordering,
                                     const std::shared_ptr<cylon::CylonContext> &ctx,
                                     std::unique_ptr<cudf::table> &split_points_table) {

  std::vector<std::unique_ptr<cudf::table>> gathered_tables;
  auto sample_tv = sample_tbl->view();
  bool gather_from_root = false;
  RETURN_CYLON_STATUS_IF_FAILED(
      gcylon::net::Gather(sample_tv, splitter, gather_from_root, ctx, gathered_tables));

  if (cylon::mpi::AmIRoot(splitter, ctx)) {
    gathered_tables.insert(gathered_tables.begin() + splitter, std::move(sample_tbl));
    RETURN_CYLON_STATUS_IF_FAILED(
        DetermineSplitPoints(gathered_tables, column_orders, null_ordering, ctx, split_points_table));
  }

  cudf::table_view source_tv;
  if (cylon::mpi::AmIRoot(splitter, ctx)) {
    source_tv = split_points_table->view();
  }

  return gcylon::net::Bcast(source_tv, splitter, ctx, split_points_table);
}

cylon::Status gcylon::SampleTableUniform(const cudf::table_view &tv,
                                         int sample_count,
                                         std::unique_ptr<cudf::table> &sampled_table) {

  if (sample_count > tv.num_rows()) {
    return cylon::Status(cylon::Code::ExecutionError,
                         "sample_count higher than the number of rows in the table");
  }

  float step = tv.num_rows() / (sample_count + 1.0);
  std::vector<int32_t> sample_indices;
  float index = step;
  for (int i = 0; i < sample_count; ++i) {
    sample_indices.push_back(index);
    index += step;
  }

  rmm::device_buffer dev_buf(sample_indices.size() * 4, rmm::cuda_stream_default);
  auto status = cudaMemcpy(dev_buf.data(), sample_indices.data(), sample_indices.size() * 4, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    return cylon::Status(cylon::Code::ExecutionError, "cudaMemcpy failed");
  }
  auto dt = cudf::data_type{cudf::type_id::INT32};
  auto clmn = std::make_unique<cudf::column>(dt, sample_indices.size(), std::move(dev_buf));
  sampled_table = cudf::gather(tv, clmn->view());

  return cylon::Status::OK();
}

cylon::Status GetSplitPointIndices(const cudf::table_view &sort_columns_tv,
                                   const cudf::table_view &split_points_tv,
                                   const std::vector<cudf::order> &column_orders,
                                   cudf::null_order null_ordering,
                                   std::vector<int32_t> &split_point_indices) {

  std::vector<cudf::null_order> column_null_orders(sort_columns_tv.num_columns(), null_ordering);

  auto split_point_indices_clmn = cudf::lower_bound(sort_columns_tv,
                                                    split_points_tv,
                                                    column_orders,
                                                    column_null_orders);

  split_point_indices.resize(split_point_indices_clmn->size(), 0);
  auto status = cudaMemcpy(split_point_indices.data(),
                           split_point_indices_clmn->view().data<uint8_t>(),
                           split_point_indices_clmn->size() * 4,
                           cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    return cylon::Status(cylon::Code::ExecutionError, "cudaMemcpy failed");
  }

  return cylon::Status::OK();
}

/**
 * resulting vector has all column indices in the table,
 * sort_column_indices occupying the first k positions
 * other column indices pushed back to the vector
 *
 * for example:
 *   if the tv has 6 columns and sort_column_indices is 1,4
 * then, resulting vector is:
 *   1,4,0,2,3,5
 *
 * @param tv
 * @param sort_column_indices
 * @return
 */
std::vector<int32_t> SortTableColumnIndices(const cudf::table_view &tv,
                                            const std::vector<int32_t> & sort_column_indices) {
  std::vector<int32_t> sort_table_column_indices;
  sort_table_column_indices.reserve(tv.num_columns());
  sort_table_column_indices.insert(sort_table_column_indices.begin(), sort_column_indices.begin(), sort_column_indices.end());
  for (int i = 0; i < tv.num_columns(); ++i) {
    if (std::find(sort_column_indices.begin(), sort_column_indices.end(), i) == sort_column_indices.end()) {
      sort_table_column_indices.push_back(i);
    }
  }
  return sort_table_column_indices;
}

/**
 * revert back the column order to the original order
 * this method is symmetrical to the method SortTableColumnIndices
 * the sorted table has column in order calculated by that method
 * this method puts columns to the original order
 *
 * @param sorted_table
 * @param sort_tbl_clm_indices
 * @return
 */
std::unique_ptr<cudf::table> RevertColumnOrder(std::unique_ptr<cudf::table> sorted_table,
                                               const std::vector<int32_t> &sort_tbl_clm_indices) {
  std::vector<std::unique_ptr<cudf::column>> columns = sorted_table->release();
  std::vector<std::unique_ptr<cudf::column>> columns2(columns.size());
  for (int i = columns.size() - 1; i >= 0; --i) {
    columns2[sort_tbl_clm_indices[i]] = std::move(columns[i]);
  }

  return std::make_unique<cudf::table>(std::move(columns2));
}

cylon::Status gcylon::DistributedSort(const cudf::table_view &tv,
                                      const std::vector<int32_t> &sort_column_indices,
                                      const std::vector<cudf::order> &column_orders,
                                      const std::shared_ptr<cylon::CylonContext> &ctx,
                                      std::unique_ptr<cudf::table> &sorted_table,
                                      bool nulls_after,
                                      const int sort_root) {

  if (sort_column_indices.size() > tv.num_columns()) {
    return cylon::Status(cylon::Code::ValueError,
                         "number of values in sort_column_indices can not larger than the number of columns");
  } else if (sort_column_indices.size() != column_orders.size()) {
    return cylon::Status(cylon::Code::ValueError,
                         "sizes of sort_column_indices and column_orders must match");
  }

  // sample_count = number_of_workers * sampling_ratio
  int sampling_ratio = GetSamplingRatio(ctx->GetWorldSize());

  // get a table_view with sort_columns in the first k positions
  // since cudf::sort method expects all columns listed as sort keys
  // we perform sorting on this table, and revert back to the original at the end
  auto sort_tbl_clm_indices = SortTableColumnIndices(tv, sort_column_indices);
  auto tv_for_sorting = tv.select(sort_tbl_clm_indices);

  std::vector<cudf::order> column_orders_all(tv.num_columns(), SortOrder(column_orders));
  for (int i = 0; i < sort_column_indices.size(); ++i) {
    column_orders_all[i] = column_orders[i];
  }

  cudf::null_order null_ordering = nulls_after ? cudf::null_order::AFTER : cudf::null_order::BEFORE;
  std::vector<cudf::null_order> column_null_orders(tv.num_columns(), null_ordering);

  // first perform local sort
  auto initial_sorted_tbl = cudf::sort(tv_for_sorting, column_orders_all, column_null_orders);
  auto initial_sorted_tv = initial_sorted_tbl->view();

  // get sort columns as a separate table_view
  auto key_column_indices = monotonicVector(sort_column_indices.size());
  auto sort_columns_tv = initial_sorted_tbl->select(key_column_indices);

  // sample the sorted table with sort columns and create a table
  int sample_count = ctx->GetWorldSize() * sampling_ratio;
  if (sample_count > sort_columns_tv.num_rows()) {
    sample_count = sort_columns_tv.num_rows();
  }
  std::unique_ptr<cudf::table> sample_tbl;
  RETURN_CYLON_STATUS_IF_FAILED(
      gcylon::SampleTableUniform(sort_columns_tv, sample_count, sample_tbl));

  // get split points
  std::unique_ptr<cudf::table> split_points_table;
  RETURN_CYLON_STATUS_IF_FAILED(
      gcylon::GetSplitPoints(std::move(sample_tbl),
                             sort_root,
                             column_orders,
                             null_ordering,
                             ctx,
                             split_points_table));
  auto split_points_tv = split_points_table->view();

  // calculate split indices on the initial sorted table
  std::vector<int32_t> split_point_indices;
  RETURN_CYLON_STATUS_IF_FAILED(
      GetSplitPointIndices(sort_columns_tv,
                           split_points_tv,
                           column_orders,
                           null_ordering,
                           split_point_indices));

  // perform all to all on split_points
  std::vector<std::unique_ptr<cudf::table>> received_tables;
  // add first and the last row index to the split indices
  split_point_indices.insert(split_point_indices.begin(), 0);
  split_point_indices.push_back(initial_sorted_tv.num_rows());
  RETURN_CYLON_STATUS_IF_FAILED(
      gcylon::net::AllToAll(initial_sorted_tv, split_point_indices, ctx, received_tables));

  // delete initial_sorted_tbl to reclaim the memory
  initial_sorted_tbl.reset();

  column_null_orders.resize(key_column_indices.size());
  std::unique_ptr<cudf::table> sorted_tbl;
  RETURN_CYLON_STATUS_IF_FAILED(
    MergeOrSortTables(received_tables, column_orders, null_ordering, sorted_tbl));
  sorted_table = RevertColumnOrder(std::move(sorted_tbl), sort_tbl_clm_indices);
  return cylon::Status::OK();
}
