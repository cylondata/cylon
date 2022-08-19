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

#include <memory>
#include <algorithm>

#include <arrow/compute/api.h>

#include <cylon/table.hpp>
#include <cylon/ctx/arrow_memory_pool_utils.hpp>
#include <cylon/util/macros.hpp>
#include <cylon/util/arrow_utils.hpp>
#include <cylon/scalar.hpp>

namespace cylon {

static constexpr int64_t kZero = 0;

/**
 * Slice the part of table to create a single table
 * @param table, offset and length
 * @return new sliced table
 */
Status Slice(const std::shared_ptr<Table> &in, int64_t offset, int64_t length,
             std::shared_ptr<Table> *out) {
  const auto &ctx = in->GetContext();
  const auto &in_table = in->get_table();

  std::shared_ptr<arrow::Table> out_table;
  if (!in->Empty()) {
    out_table = in_table->Slice(offset, length);
  } else {
    out_table = in_table;
  }
  return Table::FromArrowTable(ctx, std::move(out_table), *out);
}

/**
 * DistributedSlice the part of table to create a single table
 * @param table, global_offset and global_length
 * @return new sliced table
 */
Status distributed_slice_impl(const std::shared_ptr<Table> &in,
                              int64_t global_offset,
                              int64_t global_length,
                              int64_t *partition_lengths,
                              std::shared_ptr<Table> *out) {
  const auto &ctx = in->GetContext();
  std::shared_ptr<cylon::Column> partition_len_col;

  if (partition_lengths == nullptr) {
    const auto &num_row_scalar = Scalar::Make(arrow::MakeScalar(in->Rows()));
    RETURN_CYLON_STATUS_IF_FAILED(ctx->GetCommunicator()
                                      ->Allgather(num_row_scalar, &partition_len_col));

    partition_lengths =
        const_cast<int64_t *>(std::static_pointer_cast<arrow::Int64Array>(partition_len_col->data())
            ->raw_values());
  }

  int64_t rank = ctx->GetRank();
  int64_t prefix_sum = std::accumulate(partition_lengths, partition_lengths + rank, kZero);
  int64_t this_length = *(partition_lengths + rank);
  assert(this_length == in->Rows());

  int64_t local_offset = std::max(kZero, std::min(global_offset - prefix_sum, this_length));
  int64_t local_length =
      std::min(this_length, std::max(global_offset + global_length - prefix_sum, kZero))
          - local_offset;

  return Slice(in, local_offset, local_length, out);
}

Status DistributedSlice(const std::shared_ptr<Table> &in,
                        int64_t offset,
                        int64_t length,
                        std::shared_ptr<Table> *out) {
  return distributed_slice_impl(in, offset, length, nullptr, out);
}

/**
 * Head the part of table to create a single table with specific number of rows
 * @param tables, number of rows
 * @return new table
 */
Status Head(const std::shared_ptr<Table> &table, int64_t num_rows,
            std::shared_ptr<Table> *output) {
  if (num_rows >= 0) {
    return Slice(table, 0, num_rows, output);
  } else
    return {Code::Invalid, "Number of head rows should be >=0"};
}

Status DistributedHead(const std::shared_ptr<Table> &table, int64_t num_rows,
                       std::shared_ptr<Table> *output) {

  std::shared_ptr<arrow::Table> in_table = table->get_table();

  if (num_rows >= 0) {
    return distributed_slice_impl(table, 0, num_rows, nullptr, output);
  } else {
    return {Code::Invalid, "Number of head rows should be >=0"};
  }
}

/**
 * Tail the part of table to create a single table with specific number of rows
 * @param tables, number of rows
 * @return new table
 */
Status Tail(const std::shared_ptr<Table> &table, int64_t num_rows,
            std::shared_ptr<Table> *output) {

  std::shared_ptr<arrow::Table> in_table = table->get_table();
  const int64_t table_size = in_table->num_rows();

  if (num_rows >= 0) {
    return Slice(table, table_size - num_rows, num_rows, output);
  } else {
    return {Code::Invalid, "Number of tailed rows should be >=0"};
  }
}

Status DistributedTail(const std::shared_ptr<Table> &table, int64_t num_rows,
                       std::shared_ptr<Table> *output) {
  if (num_rows >= 0) {
    const auto &ctx = table->GetContext();
    std::shared_ptr<cylon::Column> partition_len_col;
    const auto &num_row_scalar = Scalar::Make(arrow::MakeScalar(table->Rows()));
    RETURN_CYLON_STATUS_IF_FAILED(ctx->GetCommunicator()
                                      ->Allgather(num_row_scalar, &partition_len_col));
    assert(ctx->GetWorldSize() == partition_len_col->length());
    auto *partition_lengths =
        std::static_pointer_cast<arrow::Int64Array>(partition_len_col->data())
            ->raw_values();

    int64_t dist_length =
        std::accumulate(partition_lengths, partition_lengths + ctx->GetWorldSize(), kZero);

    return distributed_slice_impl(table,
                                  dist_length - num_rows,
                                  num_rows,
                                  const_cast <int64_t *> (partition_lengths),
                                  output);
  } else {
    return {Code::Invalid, "Number of tailed rows should be >=0"};
  }
}

}