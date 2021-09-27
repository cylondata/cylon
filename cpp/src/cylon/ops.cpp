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

#include <glog/logging.h>

#include <cylon/table.hpp>
#include <cylon/ops.hpp>
#include <cylon/net/mpi/mpi_operations.hpp>

namespace cylon {

/**
 * A utility function for finding splits. Since we are calculating the splits based on the input
 * we are using the input data sizes and the cache sizes to calculate the splits. We need to improve the op framework
 * to calculate the splits at the targets. But this will be a good approximation assuming we are doing a
 * good partitioning.
 * @param ctx
 * @param left left table
 * @param right right table
 * @param left_column_indexes vector of column indexes we are working with
 * @param right_column_indexes vector of column indexes we are working with
 * @param left_splits return left splits
 * @param right_splits return right splits
 */
Status Get_Splits(const std::shared_ptr <cylon::CylonContext> &ctx,
                  const std::shared_ptr <cylon::Table> &left,
                  const std::shared_ptr <cylon::Table> &right,
                 const std::vector<int> &left_column_indexes,
                 const std::vector<int> &right_column_indexes,
                 int *left_splits,
                 int *right_splits) {
  std::array<int64_t, 2> left_sizes = cylon::util::GetBytesAndElements(left->get_table(), left_column_indexes);
  std::array<int64_t, 2> right_sizes = cylon::util::GetBytesAndElements(right->get_table(), right_column_indexes);
  std::array<int64_t, 4> all = {left_sizes[0], left_sizes[1], right_sizes[0], right_sizes[1]};
  std::array<int64_t, 4> totals;
  cylon::Status status = cylon::mpi::AllReduce(all.data(), totals.data(), 4, cylon::Int64(), cylon::net::SUM);
  RETURN_CYLON_STATUS_IF_FAILED(status);
  *left_splits = cylon::util::GetNumberSplitsToFitInCache(totals[1], totals[0], ctx->GetWorldSize());
  *right_splits = cylon::util::GetNumberSplitsToFitInCache(totals[3], totals[2], ctx->GetWorldSize());
  return Status::OK();
}

Status JoinOperation(const std::shared_ptr <cylon::CylonContext> &ctx,
                     const std::shared_ptr <cylon::Table> &left,
                     const std::shared_ptr <cylon::Table> &right,
                         const cylon::join::config::JoinConfig &join_config,
                         std::shared_ptr <cylon::Table> &out) {
  const cylon::ResultsCallback &callback = [&](int tag, const std::shared_ptr <cylon::Table> &table) {
    out = table;
  };

  int left_splits = 1;
  int right_splits = 1;
  auto status = Get_Splits(ctx, left, right, join_config.GetLeftColumnIdx(), join_config.GetRightColumnIdx(), &left_splits, &right_splits);
  RETURN_CYLON_STATUS_IF_FAILED(status);
  LOG(INFO) << "Left splits - " << left_splits << " Right splits - " << right_splits;
  const auto &part_config = cylon::PartitionOpConfig(ctx->GetWorldSize(), {0});
  const auto &dist_join_config = cylon::DisJoinOpConfig(part_config, join_config, left_splits, right_splits);
  auto op = cylon::DisJoinOP(ctx, left->get_table()->schema(), 0, callback, dist_join_config);
  op.InsertTable(100, left);
  op.InsertTable(200, right);
  auto execution = op.GetExecution();
  execution->WaitForCompletion();
  return cylon::Status::OK();
}

Status UnionOperation(const std::shared_ptr <cylon::CylonContext> &ctx,
                      const std::shared_ptr <cylon::Table> &left,
                      const std::shared_ptr <cylon::Table> &right,
                          std::shared_ptr <cylon::Table> &out) {
  const cylon::ResultsCallback &callback = [&](int tag, const std::shared_ptr <cylon::Table> &table) {
    out = table;
  };

  std::vector<int> left_column_indexes(left->Columns());
  std::vector<int> right_column_indexes(right->Columns());
  std::iota (std::begin(left_column_indexes), std::end(left_column_indexes), 0);
  std::iota (std::begin(right_column_indexes), std::end(right_column_indexes), 0);
  int left_splits = 1;
  int right_splits = 1;
  auto status = Get_Splits(ctx, left, right, left_column_indexes, right_column_indexes, &left_splits, &right_splits);
  if (!status.is_ok()) {
    LOG(INFO) << "Couldn't find the splits";
    return status;
  }
  LOG(INFO) << "Left splits - " << left_splits << " Right splits - " << right_splits;
  cylon::DisSetOpConfig unionOpConfig(left_splits, right_splits);
  auto op = cylon::DisSetOp(ctx, left->get_table()->schema(), 0, callback, unionOpConfig, cylon::kernel::UNION);
  op.InsertTable(100, left);
  op.InsertTable(200, right);
  auto execution = op.GetExecution();
  execution->WaitForCompletion();
  return cylon::Status::OK();
}

Status SubtractOperation(const std::shared_ptr <cylon::CylonContext> &ctx,
                         const std::shared_ptr <cylon::Table> &left,
                         const std::shared_ptr <cylon::Table> &right,
                             std::shared_ptr <cylon::Table> &out) {
  const cylon::ResultsCallback &callback = [&](int tag, const std::shared_ptr <cylon::Table> &table) {
    out = table;
  };

  std::vector<int> left_column_indexes(left->Columns());
  std::vector<int> right_column_indexes(right->Columns());
  std::iota (std::begin(left_column_indexes), std::end(left_column_indexes), 0);
  std::iota (std::begin(right_column_indexes), std::end(right_column_indexes), 0);
  int left_splits = 1;
  int right_splits = 1;
  auto status = Get_Splits(ctx, left, right, left_column_indexes, right_column_indexes, &left_splits, &right_splits);
  if (!status.is_ok()) {
    LOG(INFO) << "Couldn't find the splits";
    return status;
  }
  LOG(INFO) << "Left splits - " << left_splits << " Right splits - " << right_splits;
  cylon::DisSetOpConfig unionOpConfig(left_splits, right_splits);
  auto op = cylon::DisSetOp(ctx, left->get_table()->schema(), 0, callback, unionOpConfig, cylon::kernel::SUBTRACT);
  op.InsertTable(100, left);
  op.InsertTable(200, right);
  auto execution = op.GetExecution();
  execution->WaitForCompletion();
  return cylon::Status::OK();
}

Status IntersectOperation(const std::shared_ptr <cylon::CylonContext> &ctx,
                          const std::shared_ptr <cylon::Table> &left,
                          const std::shared_ptr <cylon::Table> &right,
                              std::shared_ptr <cylon::Table> &out) {
  const cylon::ResultsCallback &callback = [&](int tag, const std::shared_ptr <cylon::Table> &table) {
    out = table;
  };

  std::vector<int> left_column_indexes(left->Columns());
  std::vector<int> right_column_indexes(right->Columns());
  std::iota (std::begin(left_column_indexes), std::end(left_column_indexes), 0);
  std::iota (std::begin(right_column_indexes), std::end(right_column_indexes), 0);
  int left_splits = 1;
  int right_splits = 1;
  auto status = Get_Splits(ctx, left, right, left_column_indexes, right_column_indexes, &left_splits, &right_splits);
  if (!status.is_ok()) {
    LOG(INFO) << "Couldn't find the splits";
    return status;
  }
  LOG(INFO) << "Left splits - " << left_splits << " Right splits - " << right_splits;
  cylon::DisSetOpConfig unionOpConfig(left_splits, right_splits);
  auto op = cylon::DisSetOp(ctx, left->get_table()->schema(), 0, callback, unionOpConfig, cylon::kernel::INTERSECT);
  op.InsertTable(100, left);
  op.InsertTable(200, right);
  auto execution = op.GetExecution();
  execution->WaitForCompletion();
  return cylon::Status::OK();
}

} // namespace cylon
