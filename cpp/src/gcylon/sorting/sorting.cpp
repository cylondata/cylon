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
#include <gcylon/utils/util.hpp>

#include <cudf/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/sorting.hpp>
#include <cudf/merge.hpp>
#include <cudf/copying.hpp>
#include <cudf/search.hpp>
#include <rmm/device_buffer.hpp>
#include <cuda.h>
#include <algorithm>


cylon::Status MergeTables(const cudf::table_view &splitter_tv,
                          const std::vector<std::unique_ptr<cudf::table>> &gathered_tables,
                          const std::vector<cudf::order> &column_orders,
                          std::unique_ptr<cudf::table> &merged_table) {

  std::vector<cudf::table_view> tables_to_merge;
  tables_to_merge.reserve(gathered_tables.size() + 1);
  tables_to_merge.push_back(splitter_tv);
  for (long unsigned int i = 0; i < gathered_tables.size(); ++i) {
    tables_to_merge.push_back(gathered_tables[i]->view());
  }

  std::vector<cudf::size_type> key_cols;
  key_cols.reserve(splitter_tv.num_columns());
  for (int i = 0; i < splitter_tv.num_columns(); ++i) {
    key_cols.push_back(i);
  }

  merged_table = cudf::merge(tables_to_merge, key_cols, column_orders);
  if (merged_table == nullptr) {
    return cylon::Status(cylon::Code::ExecutionError, "merge failed");
  }

  return cylon::Status::OK();
}

cylon::Status DetermineSplitPoints(const cudf::table_view &splitter_tv,
                                   const std::vector<std::unique_ptr<cudf::table>> &gathered_tables,
                                   const std::vector<cudf::order> &column_orders,
                                   const std::shared_ptr<cylon::CylonContext> &ctx,
                                   std::unique_ptr<cudf::table> &split_points_table) {

  //todo: when the number of workers is more than 10, we can sort instead of merging
  std::unique_ptr<cudf::table> merged_table;
  RETURN_CYLON_STATUS_IF_FAILED(
      MergeTables(splitter_tv, gathered_tables, column_orders, merged_table));

  int num_split_points = ctx->GetWorldSize() - 1;
  auto merged_tv = merged_table->view();
  return gcylon::SampleTableUniform(merged_tv, num_split_points, split_points_table);
}

cylon::Status gcylon::GetSplitPoints(const cudf::table_view &sample_tv,
                                     int splitter,
                                     const std::vector<cudf::order> &column_orders,
                                     const std::shared_ptr<cylon::CylonContext> &ctx,
                                     std::unique_ptr<cudf::table> &split_points_table) {

  std::vector<std::unique_ptr<cudf::table>> gathered_tables;
  bool gather_from_root = false;
  RETURN_CYLON_STATUS_IF_FAILED(
      gcylon::net::Gather(sample_tv, splitter, gather_from_root, ctx, gathered_tables));

  if (cylon::mpi::AmIRoot(splitter, ctx)) {
    RETURN_CYLON_STATUS_IF_FAILED(
        DetermineSplitPoints(sample_tv, gathered_tables, column_orders, ctx, split_points_table));
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

  // sample_count = number_of_workers * SAMPLING_RATIO
  const int SAMPLING_RATIO = 2;

  // get a table_view with sort_columns in the first k positions
  // since cudf::sort method expects all columns listed as sort keys
  // we perform sorting on this table, and revert back to the original at the end
  std::vector<int32_t> column_indices_4s;
  column_indices_4s.reserve(tv.num_columns());
  column_indices_4s.insert(column_indices_4s.begin(), sort_column_indices.begin(), sort_column_indices.end());
  for (int i = 0; i < tv.num_columns(); ++i) {
    if (std::find(sort_column_indices.begin(), sort_column_indices.end(), i) == sort_column_indices.end()) {
      column_indices_4s.push_back(i);
    }
  }
  auto tv_for_sorting = tv.select(column_indices_4s);

  // sort unspecified columns in ASCENDING order
  std::vector<cudf::order> column_orders_all(tv.num_columns(), cudf::order::ASCENDING);
  for (int i = 0; i < sort_column_indices.size(); ++i) {
    column_orders_all[i] = column_orders[i];
  }

  cudf::null_order null_ordering = nulls_after ? cudf::null_order::AFTER : cudf::null_order::BEFORE;
  std::vector<cudf::null_order> column_null_orders(tv.num_columns(), null_ordering);

  // first perform local sort
  auto initial_sorted_tbl = cudf::sort(tv_for_sorting, column_orders_all, column_null_orders);
  auto initial_sorted_tv = initial_sorted_tbl->view();

  // get sort columns as a separate table_view
  std::vector<int32_t> key_column_indices;
  for (int i = 0; i < sort_column_indices.size(); ++i) {
    key_column_indices.push_back(i);
  }
  auto sort_columns_tv = initial_sorted_tbl->select(key_column_indices);

  // sample the sorted table with sort columns and create a table
  int sample_count = ctx->GetWorldSize() * SAMPLING_RATIO;
  if (sample_count > sort_columns_tv.num_rows()) {
    sample_count = sort_columns_tv.num_rows();
  }
  std::unique_ptr<cudf::table> sample_tbl;
  RETURN_CYLON_STATUS_IF_FAILED(
      gcylon::SampleTableUniform(sort_columns_tv, sample_count, sample_tbl));
  auto sample_tv = sample_tbl->view();

  // get split points
  std::unique_ptr<cudf::table> split_points_table;
  RETURN_CYLON_STATUS_IF_FAILED(
      gcylon::GetSplitPoints(sample_tv,
                             sort_root,
                             column_orders,
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

  // merge received tables
  std::vector<cudf::table_view> tv_to_merge{};
  tv_to_merge.reserve(received_tables.size());
  for (long unsigned int i = 0; i < received_tables.size(); i++) {
    tv_to_merge.push_back(received_tables[i]->view());
  }

  column_null_orders.resize(key_column_indices.size());
  auto sorted_tbl = cudf::merge(tv_to_merge, key_column_indices, column_orders, column_null_orders);

  // change order of columns to the original
  std::vector<std::unique_ptr<cudf::column>> columns = sorted_tbl->release();
  std::vector<std::unique_ptr<cudf::column>> columns2(columns.size());
  for (int i = columns.size() - 1; i >= 0; --i) {
    columns2[column_indices_4s[i]] = std::move(columns[i]);
  }

  sorted_table = std::make_unique<cudf::table>(std::move(columns2));
  return cylon::Status::OK();
}
