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

#include <cudf/partitioning.hpp>
#include <cudf/join.hpp>
#include <cudf/io/csv.hpp>

#include <gcylon/gtable.hpp>
#include <gcylon/gtable_api.hpp>
#include <gcylon/net/cudf_net_ops.hpp>

#include <cylon/util/macros.hpp>
#include <cylon/net/mpi/mpi_operations.hpp>
#include <cylon/repartition.hpp>

namespace gcylon {

cylon::Status Shuffle(const cudf::table_view &input_tv,
                      const std::vector<int> &columns_to_hash,
                      const std::shared_ptr<cylon::CylonContext> &ctx,
                      std::unique_ptr<cudf::table> &table_out) {

  std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> partitioned
    = cudf::hash_partition(input_tv, columns_to_hash, ctx->GetWorldSize());

  partitioned.second.push_back(input_tv.num_rows());

  RETURN_CYLON_STATUS_IF_FAILED(
    gcylon::net::AllToAll(partitioned.first->view(), partitioned.second, ctx, table_out));

  return cylon::Status::OK();
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
    auto evenly_dist_rows = cylon::DivideRowsEvenly(current_row_counts);
    rows_to_all = cylon::RowIndicesToAll(ctx->GetRank(), current_row_counts, evenly_dist_rows);
  } else {
    auto sum_of_current_rows = std::accumulate(current_row_counts.begin(), current_row_counts.end(), 0);
    auto sum_of_target_rows = std::accumulate(rows_per_worker.begin(), rows_per_worker.end(), 0);
    if (sum_of_current_rows != sum_of_target_rows) {
      return cylon::Status(cylon::Code::ValueError,
                           "Sum of target partitions does not match the sum of current partitions.");
    }
    rows_to_all = cylon::RowIndicesToAll(ctx->GetRank(), current_row_counts, rows_per_worker);
  }

  RETURN_CYLON_STATUS_IF_FAILED(
    gcylon::net::AllToAll(input_tv, rows_to_all, ctx, table_out));

  return cylon::Status::OK();
}

cylon::Status Shuffle(std::shared_ptr<GTable> &input_table,
                      const std::vector<int> &columns_to_hash,
                      std::shared_ptr<GTable> &output_table) {

  std::unique_ptr<cudf::table> table_out;
  auto ctx = input_table->GetContext();

  RETURN_CYLON_STATUS_IF_FAILED(
    Shuffle(input_table->GetCudfTable()->view(), columns_to_hash, ctx, table_out));

  RETURN_CYLON_STATUS_IF_FAILED(
    GTable::FromCudfTable(ctx, table_out, output_table));

  // set metadata for the shuffled table
  output_table->SetCudfMetadata(input_table->GetCudfMetadata());

  return cylon::Status::OK();
}

cylon::Status joinTables(const cudf::table_view &left,
                         const cudf::table_view &right,
                         const cylon::join::config::JoinConfig &join_config,
                         std::shared_ptr<cylon::CylonContext> ctx,
                         std::unique_ptr<cudf::table> &table_out) {

  if (join_config.GetAlgorithm() == cylon::join::config::JoinAlgorithm::SORT) {
    return cylon::Status(cylon::Code::NotImplemented, "SORT join is not supported on GPUs yet.");
  }

  if (join_config.GetType() == cylon::join::config::JoinType::INNER) {
    table_out = cudf::inner_join(left,
                                 right,
                                 join_config.GetLeftColumnIdx(),
                                 join_config.GetRightColumnIdx());
  } else if (join_config.GetType() == cylon::join::config::JoinType::LEFT) {
    table_out = cudf::left_join(left,
                                right,
                                join_config.GetLeftColumnIdx(),
                                join_config.GetRightColumnIdx());
  } else if (join_config.GetType() == cylon::join::config::JoinType::RIGHT) {
    table_out = cudf::left_join(right,
                                left,
                                join_config.GetRightColumnIdx(),
                                join_config.GetLeftColumnIdx());
  } else if (join_config.GetType() == cylon::join::config::JoinType::FULL_OUTER) {
    table_out = cudf::full_join(left,
                                right,
                                join_config.GetLeftColumnIdx(),
                                join_config.GetRightColumnIdx());
  }

  return cylon::Status::OK();
}


cylon::Status joinTables(std::shared_ptr<GTable> &left,
                         std::shared_ptr<GTable> &right,
                         const cylon::join::config::JoinConfig &join_config,
                         std::shared_ptr<GTable> &joined_table) {

  if (left == nullptr) {
    return cylon::Status(cylon::Code::KeyError, "Couldn't find the left table");
  } else if (right == nullptr) {
    return cylon::Status(cylon::Code::KeyError, "Couldn't find the right table");
  }

  if (join_config.GetAlgorithm() == cylon::join::config::JoinAlgorithm::SORT) {
    return cylon::Status(cylon::Code::NotImplemented, "SORT join is not supported on GPUs yet.");
  }

  std::shared_ptr<cylon::CylonContext> ctx = left->GetContext();
  std::unique_ptr<cudf::table> joined;
  RETURN_CYLON_STATUS_IF_FAILED(joinTables(left->GetCudfTable()->view(),
                                           right->GetCudfTable()->view(),
                                           join_config,
                                           ctx,
                                           joined));

  RETURN_CYLON_STATUS_IF_FAILED(
    GTable::FromCudfTable(ctx, joined, joined_table));

  // set metadata for the joined table
  joined_table->SetCudfMetadata(left->GetCudfMetadata());
  return cylon::Status::OK();
}

/**
* Similar to local join, but performs the join in a distributed fashion
 * works on tale_view objects
* @param left_table
* @param right_table
* @param join_config
* @param ctx
* @param table_out
* @return
*/
cylon::Status DistributedJoin(const cudf::table_view &left_table,
                              const cudf::table_view &right_table,
                              const cylon::join::config::JoinConfig &join_config,
                              const std::shared_ptr<cylon::CylonContext> &ctx,
                              std::unique_ptr<cudf::table> &table_out) {

  if (ctx->GetWorldSize() == 1) {
    // perform single join
    return joinTables(left_table, right_table, join_config, ctx, table_out);
  }

  std::unique_ptr<cudf::table> left_shuffled_table, right_shuffled_table;

  RETURN_CYLON_STATUS_IF_FAILED(
    Shuffle(left_table, join_config.GetLeftColumnIdx(), ctx, left_shuffled_table));

  RETURN_CYLON_STATUS_IF_FAILED(
    Shuffle(right_table, join_config.GetRightColumnIdx(), ctx, right_shuffled_table));

  RETURN_CYLON_STATUS_IF_FAILED(
    joinTables(left_shuffled_table->view(), right_shuffled_table->view(), join_config, ctx, table_out));

  return cylon::Status::OK();
}

/**
 * Similar to local join, but performs the join in a distributed fashion
 * works on GTable objects
 * @param left
 * @param right
 * @param join_config
 * @param output
 * @return
 */
cylon::Status DistributedJoin(std::shared_ptr<GTable> &left,
                              std::shared_ptr<GTable> &right,
                              const cylon::join::config::JoinConfig &join_config,
                              std::shared_ptr<GTable> &output) {

  std::shared_ptr<cylon::CylonContext> ctx = left->GetContext();
  if (ctx->GetWorldSize() == 1) {
    // perform single join
    return joinTables(left, right, join_config, output);
  }

  std::shared_ptr<GTable> left_shuffled_table, right_shuffled_table;

  RETURN_CYLON_STATUS_IF_FAILED(
    Shuffle(left, join_config.GetLeftColumnIdx(), left_shuffled_table));

  RETURN_CYLON_STATUS_IF_FAILED(
    Shuffle(right, join_config.GetRightColumnIdx(), right_shuffled_table));

  RETURN_CYLON_STATUS_IF_FAILED(
    joinTables(left_shuffled_table, right_shuffled_table, join_config, output));

  return cylon::Status::OK();
}

/**
 * write GTable to file
 * @param table
 * @param output_file
 * @return
 */
cylon::Status WriteToCsv(std::shared_ptr<GTable> &table, std::string output_file) {
  cudf::io::sink_info sink_info(output_file);
  cudf::io::csv_writer_options options =
    cudf::io::csv_writer_options::builder(sink_info, table->GetCudfTable()->view())
      .metadata(&(table->GetCudfMetadata()))
      .include_header(true);
  cudf::io::write_csv(options);
  return cylon::Status::OK();
}

/**
 * get table sizes from all workers
 * each worker size in the table_sizes[rank]
 * @param num_rows size of the table at the current worker
 * @param ctx
 * @param all_num_rows all tables sizes from all workers
 * @return
 */
cylon::Status RowCountsAllTables(int32_t num_rows,
                                 const std::shared_ptr<cylon::CylonContext> &ctx,
                                 std::vector<int32_t> &all_num_rows) {
  std::vector<int32_t> send_data(1, num_rows);
  return cylon::mpi::AllGather(send_data, ctx->GetWorldSize(), all_num_rows);
}


}// end of namespace gcylon
