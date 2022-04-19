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
#include <arrow/compute/api.h>
#include <arrow/table.h>

#include <fstream>
#include <future>
#include <memory>
#include <unordered_map>
#include <iostream>

#include <cylon/table.hpp>
#include <cylon/join/join_utils.hpp>
#include <cylon/arrow/arrow_all_to_all.hpp>
#include <cylon/arrow/arrow_comparator.hpp>
#include <cylon/arrow/arrow_types.hpp>
#include <cylon/ctx/arrow_memory_pool_utils.hpp>
#include <cylon/io/arrow_io.hpp>
#include <cylon/join/join.hpp>
#include <cylon/partition/partition.hpp>
#include <cylon/table_api_extended.hpp>
#include <cylon/thridparty/flat_hash_map/bytell_hash_map.hpp>
#include <cylon/util/arrow_utils.hpp>
#include <cylon/util/macros.hpp>
#include <cylon/util/to_string.hpp>
#include <cylon/util/arrow_utils.hpp>
#include <cylon/repartition.hpp>
#include <cylon/net/mpi/mpi_operations.hpp>
#include <cylon/serialize/table_serialize.hpp>

namespace cylon {

/**
 * creates an Arrow array based on col_idx, filtered by row_indices
 * @param ctx
 * @param table
 * @param col_idx
 * @param row_indices
 * @param array_vector
 * @return
 */
Status PrepareArray(std::shared_ptr<cylon::CylonContext> &ctx,
                    const std::shared_ptr<arrow::Table> &table, const int32_t col_idx,
                    const std::vector<int64_t> &row_indices, arrow::ArrayVector &array_vector) {
  std::shared_ptr<arrow::Array> destination_col_array;
  arrow::Status ar_status =
      cylon::util::copy_array_by_indices(row_indices,
                                         cylon::util::GetChunkOrEmptyArray(table->column(col_idx),
                                                                           0),
                                         &destination_col_array,
                                         cylon::ToArrowPool(ctx));
  if (ar_status != arrow::Status::OK()) {
    LOG(FATAL) << "Failed while copying a column to the final table from tables."
               << ar_status.ToString();
    return Status(static_cast<int>(ar_status.code()), ar_status.message());
  }
  array_vector.push_back(destination_col_array);
  return Status::OK();
}
  
static inline Status all_to_all_arrow_tables(const std::shared_ptr<CylonContext> &ctx,
                                             const std::shared_ptr<arrow::Schema> &schema,
                                             const std::vector<std::shared_ptr<arrow::Table>> &partitioned_tables,
                                             std::shared_ptr<arrow::Table> &table_out) {
  const auto &neighbours = ctx->GetNeighbours(true);
  std::vector<std::shared_ptr<arrow::Table>> received_tables;
  received_tables.reserve(neighbours.size());

  // define call back to catch the receiving tables
  ArrowCallback arrow_callback =
      [&received_tables](int source, const std::shared_ptr<arrow::Table> &table_, int reference) {
        CYLON_UNUSED(source);
        CYLON_UNUSED(reference);
        received_tables.push_back(table_);
        return true;
      };

  // doing all to all communication to exchange tables
  cylon::ArrowAllToAll all_to_all(ctx, neighbours, neighbours, ctx->GetNextSequence(),
                                  arrow_callback, schema);

  // if world size == partitions, simply send paritions based on index
  const int world_size = ctx->GetWorldSize(),
      num_partitions = (int) partitioned_tables.size(),
      rank = ctx->GetRank();
  if (world_size == num_partitions) {
    for (int i = 0; i < num_partitions; i++) {
      if (i != rank) {
        all_to_all.insert(partitioned_tables[i], i);
      } else {
        received_tables.push_back(partitioned_tables[i]);
      }
    }
  } else {  // divide partitions to world_size potions and send accordingly
    for (int i = 0; i < num_partitions; i++) {
      int target = i * world_size / num_partitions;
      if (target != rank) {
        all_to_all.insert(partitioned_tables[i], target);
      } else {
        received_tables.push_back(partitioned_tables[i]);
      }
    }
  }

  // now complete the communication
  all_to_all.finish();
  while (!all_to_all.isComplete()) {
  }
  all_to_all.close();

  /*  // now clear locally partitioned tables
  partitioned_tables.clear();*/

  // now we have the final set of tables
//  LOG(INFO) << "Concatenating tables, Num of tables :  " << received_tables.size();
  CYLON_ASSIGN_OR_RAISE(auto concat, arrow::ConcatenateTables(received_tables))
//  LOG(INFO) << "Done concatenating tables, rows :  " << concat->num_rows();

  CYLON_ASSIGN_OR_RAISE(table_out, concat->CombineChunks(cylon::ToArrowPool(ctx)))
  return Status::OK();
}

// entries from each RANK are separated
static inline Status all_to_all_arrow_tables_separated_arrow_table(const std::shared_ptr<CylonContext> &ctx,
                                             const std::shared_ptr<arrow::Schema> &schema,
                                             const std::vector<std::shared_ptr<arrow::Table>> &partitioned_tables,
                                             std::vector<std::shared_ptr<arrow::Table>> &received_tables) {
  const auto &neighbours = ctx->GetNeighbours(true);
  received_tables.resize(ctx->GetWorldSize());

  // define call back to catch the receiving tables
  ArrowCallback arrow_callback =
      [&received_tables, &ctx](int source, const std::shared_ptr<arrow::Table> &table_, int reference) {
        CYLON_UNUSED(reference);
        received_tables[source] = table_;
        return true;
      };

  // doing all to all communication to exchange tables
  cylon::ArrowAllToAll all_to_all(ctx, neighbours, neighbours, ctx->GetNextSequence(),
                                  arrow_callback, schema);

  // if world size == partitions, simply send paritions based on index
  const int world_size = ctx->GetWorldSize(),
      num_partitions = (int) partitioned_tables.size(),
      rank = ctx->GetRank();
  if (world_size == num_partitions) {
    for (int i = 0; i < num_partitions; i++) {
      if (i != rank) {
        all_to_all.insert(partitioned_tables[i], i);
      } else {
        received_tables[i] = partitioned_tables[i];
      }
    }
  } else {  // divide partitions to world_size potions and send accordingly
    for (int i = 0; i < num_partitions; i++) {
      int target = i * world_size / num_partitions;
      if (target != rank) {
        all_to_all.insert(partitioned_tables[i], target);
      } else {
        received_tables[i] = partitioned_tables[i];
      }
    }
  }

  // now complete the communication
  all_to_all.finish();
  while (!all_to_all.isComplete()) {
  }
  all_to_all.close();

  return Status::OK();
}

static inline Status all_to_all_arrow_tables_separated_cylon_table(const std::shared_ptr<CylonContext> &ctx,
                                             const std::shared_ptr<arrow::Schema> &schema,
                                             const std::vector<std::shared_ptr<arrow::Table>> &partitioned_tables,
                                             std::vector<std::shared_ptr<Table>> &table_out) {
  std::vector<std::shared_ptr<arrow::Table>> received_tables;
  all_to_all_arrow_tables_separated_arrow_table(ctx, schema, partitioned_tables, received_tables);

  table_out.reserve(received_tables.size() - 1);
  for(int i = 0; i < received_tables.size(); i++) {
    if(received_tables[i]->num_rows() > 0) {
      CYLON_ASSIGN_OR_RAISE(auto arrow_tb, received_tables[i]->CombineChunks(cylon::ToArrowPool(ctx)));
      auto temp = std::make_shared<Table>(ctx, std::move(arrow_tb));
      table_out.push_back(temp);
    }
  }

  return Status::OK();
}

/**
 * output rows order by rank number
 */
static inline Status all_to_all_arrow_tables_preserve_order(const std::shared_ptr<CylonContext> &ctx,
                                                            const std::shared_ptr<arrow::Schema> &schema,
                                                            const std::vector<std::shared_ptr<arrow::Table>> &partitioned_tables,
                                                            std::shared_ptr<arrow::Table> &table_out) {
  std::vector<std::shared_ptr<arrow::Table>> tables;
  RETURN_CYLON_STATUS_IF_FAILED(all_to_all_arrow_tables_separated_arrow_table(ctx, schema, partitioned_tables, tables));
  LOG(INFO) << "Concatenating tables, Num of tables :  " << tables.size();
  CYLON_ASSIGN_OR_RAISE(table_out, arrow::ConcatenateTables(tables));
  LOG(INFO) << "Done concatenating tables, rows :  " << table_out->num_rows();

  return Status::OK();
}

template<typename T>
// T is int32_t or const std::vector<int32_t>&
static inline Status shuffle_table_by_hashing(const std::shared_ptr<CylonContext> &ctx,
                                              const std::shared_ptr<Table> &table,
                                              const T &hash_column,
                                              std::shared_ptr<arrow::Table> &table_out) {
  // partition the tables locally
  std::vector<uint32_t> outPartitions, counts;
  int no_of_partitions = ctx->GetWorldSize();
  RETURN_CYLON_STATUS_IF_FAILED(
      MapToHashPartitions(table, hash_column, no_of_partitions, outPartitions, counts));

  std::vector<std::shared_ptr<arrow::Table>> partitioned_tables;
  RETURN_CYLON_STATUS_IF_FAILED(
      Split(table, no_of_partitions, outPartitions, counts, partitioned_tables));

  std::shared_ptr<arrow::Schema> schema = table->get_table()->schema();
  // we are going to free if retain is set to false
  if (!table->IsRetain()) {
    const_cast<std::shared_ptr<Table> &>(table).reset();
  }

  return all_to_all_arrow_tables(ctx, schema, partitioned_tables, table_out);
}

template<typename T>
// T is int32_t or const std::vector<int32_t>&
static inline Status shuffle_two_tables_by_hashing(const std::shared_ptr<cylon::CylonContext> &ctx,
                                                   const std::shared_ptr<Table> &left_table,
                                                   const T &left_hash_column,
                                                   const std::shared_ptr<Table> &right_table,
                                                   const T &right_hash_column,
                                                   std::shared_ptr<arrow::Table> &left_table_out,
                                                   std::shared_ptr<arrow::Table> &right_table_out) {
  LOG(INFO) << "Shuffling two tables with total rows : "
            << left_table->Rows() + right_table->Rows();
  auto t1 = std::chrono::high_resolution_clock::now();
  RETURN_CYLON_STATUS_IF_FAILED(
      shuffle_table_by_hashing(ctx, left_table, left_hash_column, left_table_out));

  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Left shuffle time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  RETURN_CYLON_STATUS_IF_FAILED(
      shuffle_table_by_hashing(ctx, right_table, right_hash_column, right_table_out));

  auto t3 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Right shuffle time : "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

  return Status::OK();
}

Status FromCSV(const std::shared_ptr<CylonContext> &ctx, const std::string &path,
               std::shared_ptr<Table> &tableOut, const cylon::io::config::CSVReadOptions &options) {
  arrow::Result<std::shared_ptr<arrow::Table>> result = cylon::io::read_csv(ctx, path, options);
  if (result.ok()) {
    std::shared_ptr<arrow::Table> &table = result.ValueOrDie();
    if (table->column(0)->chunks().size() > 1) {
      const auto &combine_res = table->CombineChunks(ToArrowPool(ctx));
      if (!combine_res.ok()) {
        return Status(static_cast<int>(combine_res.status().code()),
                      combine_res.status().message());
      }
      table = combine_res.ValueOrDie();
    }
    // slice the table if required
    if (options.IsSlice() && ctx->GetWorldSize() > 1) {
      int32_t rows_per_worker = table->num_rows() / ctx->GetWorldSize();
      int32_t remainder = table->num_rows() % ctx->GetWorldSize();

      // first few workers will balance out the remainder
      int32_t balancer = 0;
      if (remainder != 0 && ctx->GetRank() < remainder) {
        balancer = 1;
      }

      // start index should offset the balanced out rows by previous workers
      int32_t offset = ctx->GetRank();
      if (ctx->GetRank() >= remainder) {
        offset = remainder;
      }

      int32_t starting_index = (ctx->GetRank() * rows_per_worker) + offset;
      table = table->Slice(starting_index, rows_per_worker + balancer);
    }
    tableOut = std::make_shared<Table>(ctx, table);
    return Status::OK();
  }
  return Status(Code::IOError, result.status().message());
}

Status Table::FromArrowTable(const std::shared_ptr<CylonContext> &ctx,
                             std::shared_ptr<arrow::Table> table,
                             std::shared_ptr<Table> &tableOut) {
  RETURN_CYLON_STATUS_IF_FAILED(tarrow::CheckSupportedTypes(table));
  tableOut = std::make_shared<Table>(ctx, std::move(table));
  return Status::OK();
}

Status Table::FromColumns(const std::shared_ptr<CylonContext> &ctx,
                          const std::vector<std::shared_ptr<Column>> &columns,
                          const std::vector<std::string> &column_names,
                          std::shared_ptr<Table> &tableOut) {
  arrow::SchemaBuilder schema_builder;
  arrow::ArrayVector arrays;

  if (columns.size() != column_names.size()){
    return {Code::Invalid, "number of columns != number of column names"};
  }

  for (size_t i = 0; i < columns.size(); i++) {
    const auto &data_type = columns[i]->type();
    const auto &field = arrow::field(column_names[i], tarrow::ToArrowType(data_type));
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(schema_builder.AddField(field));
    arrays.push_back(columns[i]->data());
  }

  CYLON_ASSIGN_OR_RAISE(auto schema, schema_builder.Finish());

  auto table = arrow::Table::Make(std::move(schema), arrays);

  RETURN_CYLON_STATUS_IF_FAILED(tarrow::CheckSupportedTypes(table));

  return Table::FromArrowTable(ctx, std::move(table), tableOut);
}

Status WriteCSV(const std::shared_ptr<Table> &table, const std::string &path,
                const cylon::io::config::CSVWriteOptions &options) {
  std::ofstream out_csv;
  out_csv.open(path);
  Status status = table->PrintToOStream(
      0, table->get_table()->num_columns(), 0, table->get_table()->num_rows(), out_csv,
      options.GetDelimiter(), options.IsOverrideColumnNames(), options.GetColumnNames());
  out_csv.close();
  return status;
}

int Table::Columns() const { return table_->num_columns(); }

std::vector<std::string> Table::ColumnNames() { return table_->ColumnNames(); }

int64_t Table::Rows() const { return table_->num_rows(); }

bool Table::Empty() const { return table_->num_rows() == 0; }

void Table::Print() const { Print(0, this->Rows(), 0, this->Columns()); }

void Table::Print(int row1, int row2, int col1, int col2) const {
  PrintToOStream(col1, col2, row1, row2, std::cout);
}

Status Merge(const std::vector<std::shared_ptr<cylon::Table>> &ctables,
             std::shared_ptr<Table> &tableOut) {
  if (!ctables.empty()) {
    std::vector<std::shared_ptr<arrow::Table>> tables;
    tables.reserve(ctables.size());
    for (const auto &t: ctables) {
      if (t->Rows()) {
        std::shared_ptr<arrow::Table> arrow;
        t->ToArrowTable(arrow);
        tables.push_back(std::move(arrow));
      }
    }

    const auto &ctx = ctables[0]->GetContext();
    const auto &concat_res = arrow::ConcatenateTables(tables);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(concat_res.status());

    auto combined_res = concat_res.ValueOrDie()->CombineChunks(cylon::ToArrowPool(ctx));
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(combined_res.status());

    tableOut = std::make_shared<cylon::Table>(ctx, combined_res.ValueOrDie());
    return Status::OK();
  } else {
    return Status(Code::Invalid, "empty vector passed onto merge");
  }
}

Status Sort(const std::shared_ptr<Table> &table, int sort_column,
            std::shared_ptr<cylon::Table> &out, bool ascending) {
  std::shared_ptr<arrow::Table> sorted_table;
  const auto &table_ = table->get_table();
  const auto &ctx = table->GetContext();
  auto pool = cylon::ToArrowPool(ctx);

  // if num_rows is 0 or 1, we dont need to sort
  if (table->Rows() < 2) {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::Duplicate(table_, pool, sorted_table));
  }

  RETURN_CYLON_STATUS_IF_ARROW_FAILED(
      util::SortTable(table_, sort_column, pool, sorted_table, ascending));
  return Table::FromArrowTable(ctx, sorted_table, out);
}

Status Sort(const std::shared_ptr<Table> &table, const std::vector<int32_t> &sort_columns,
            std::shared_ptr<cylon::Table> &out, bool ascending) {
  const std::vector<bool> sort_direction(sort_columns.size(), ascending);
  return Sort(table, sort_columns, out, sort_direction);
}

Status Sort(const std::shared_ptr<Table> &table, const std::vector<int32_t> &sort_columns,
            std::shared_ptr<cylon::Table> &out, const std::vector<bool> &sort_direction) {
  // if single index sort is passed
  if (sort_columns.size() == 1) {
    return Sort(table, sort_columns[0], out, sort_direction[0]);
  }

  std::shared_ptr<arrow::Table> sorted_table;
  auto table_ = table->get_table();
  const auto &ctx = table->GetContext();
  auto pool = cylon::ToArrowPool(ctx);

  // if num_rows is 0 or 1, we dont need to sort
  if (table->Rows() < 2) {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::Duplicate(table_, pool, sorted_table));
  }

  RETURN_CYLON_STATUS_IF_ARROW_FAILED(cylon::util::SortTableMultiColumns(table_,
                                                                         sort_columns,
                                                                         pool,
                                                                         sorted_table,
                                                                         sort_direction));
  return Table::FromArrowTable(ctx, sorted_table, out);
}

Status SampleTableUniform(const std::shared_ptr<Table> &local_sorted,
                          int num_samples, std::vector<int32_t> sort_columns,
                          std::shared_ptr<Table> &sample_result,
                          const std::shared_ptr<CylonContext> &ctx) {
  auto pool = cylon::ToArrowPool(ctx);
  
  CYLON_ASSIGN_OR_RAISE(auto local_sorted_selected_cols, local_sorted->get_table()->SelectColumns(sort_columns));

  if (local_sorted->Rows() == 0 || num_samples == 0) {
    std::shared_ptr<arrow::Table> output;
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::CreateEmptyTable(
        local_sorted_selected_cols->schema(), &output, pool));
    sample_result = std::make_shared<Table>(ctx, std::move(output));
    return Status::OK();
  }

  float step = local_sorted->Rows() / (num_samples + 1.0);
  float acc = step;
  arrow::Int64Builder filter(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(filter.Reserve(num_samples));

  for (int i = 0; i < num_samples; i++) {
    filter.UnsafeAppend(acc);
    acc += step;
  }

  CYLON_ASSIGN_OR_RAISE(auto take_arr, filter.Finish());
  CYLON_ASSIGN_OR_RAISE(
      auto take_res,
      (arrow::compute::Take(local_sorted_selected_cols, take_arr)));
  sample_result = std::make_shared<Table>(ctx, take_res.table());

  return Status::OK();
}

template <typename T>
static int CompareRows(const std::vector<std::unique_ptr<T>> &comparators,
                       int64_t idx_a,
                       int64_t idx_b) {
  int sz = comparators.size();
  if (std::is_same<T, cylon::DualArrayIndexComparator>::value) {
    idx_b |= (int64_t)1 << 63;
  }
  for (int i = 0; i < sz; i++) {
    int result = comparators[i]->compare(idx_a, idx_b);
    if (result == 0) continue;
    return result;
  }
  return 0;
}

Status MergeSortedTable(const std::vector<std::shared_ptr<Table>> &tables,
                        const std::vector<int> &sort_columns,
                        const std::vector<bool> &sort_orders,
                        std::shared_ptr<Table> &out, bool use_merge) {
  std::shared_ptr<Table> concatenated;
  std::vector<int64_t> table_indices(tables.size()),
      table_end_indices(tables.size());
  int acc = 0;
  for (int i = 0; i < table_indices.size(); i++) {
    table_indices[i] = acc;
    acc += tables[i]->Rows();
    table_end_indices[i] = acc;
  }

  RETURN_CYLON_STATUS_IF_FAILED(Merge(tables, concatenated));

  if(!use_merge) {
    return Sort(concatenated, sort_columns, out, sort_orders);
  }

  std::unique_ptr<TableRowIndexEqualTo> equal_to;
  RETURN_CYLON_STATUS_IF_FAILED(TableRowIndexEqualTo::Make(
      concatenated->get_table(), sort_columns, &equal_to, sort_orders));

  auto comp = [&](int a, int b) {  // a and b are index of table in `tables`
    int64_t a_idx = table_indices[a], b_idx = table_indices[b];
    return equal_to->compare(a_idx, b_idx) > 0;
  };

  std::priority_queue<int, std::vector<int>, decltype(comp)> pq(comp);

  for (int i = 0; i < tables.size(); i++) {
    if (table_indices[i] < table_end_indices[i]) {
      pq.push(i);
    }
  }

  auto ctx = concatenated->GetContext();
  auto pool = cylon::ToArrowPool(ctx);
  arrow::Int64Builder filter(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(filter.Reserve(concatenated->Rows()));

  std::vector<int> temp_v;

  while (!pq.empty()) {
    int t = pq.top();
    pq.pop();
    // std::cout<<table_indices[t]<<std::endl;
    temp_v.push_back(table_indices[t]);
    filter.UnsafeAppend(table_indices[t]);
    table_indices[t] += 1;
    if (table_indices[t] < table_end_indices[t]) {
      pq.push(t);
    }
  }

  CYLON_ASSIGN_OR_RAISE(auto take_arr, filter.Finish());
  CYLON_ASSIGN_OR_RAISE(
      auto take_res,
      (arrow::compute::Take(concatenated->get_table(), take_arr)));

  out = std::make_shared<Table>(ctx, take_res.table());

  return Status::OK();
}

Status DetermineSplitPoints(
    const std::vector<std::shared_ptr<Table>> &gathered_tables_include_root,
    const std::vector<bool> &sort_orders,
    std::shared_ptr<Table> &split_points,
    const std::shared_ptr<CylonContext> &ctx) {
  std::shared_ptr<Table> merged_table;

  std::vector<int32_t> sort_columns(sort_orders.size());
  std::iota(sort_columns.begin(), sort_columns.end(), 0);

  RETURN_CYLON_STATUS_IF_FAILED(MergeSortedTable(
      gathered_tables_include_root, sort_columns, sort_orders, merged_table, true));

  int num_split_points =
      std::min(merged_table->Rows(), (int64_t)ctx->GetWorldSize() - 1);

  return SampleTableUniform(merged_table, num_split_points, sort_columns, split_points, ctx);
}

Status GetSplitPoints(std::shared_ptr<Table> &sample_result,
                      const std::vector<bool> &sort_orders,
                      int num_split_points,
                      std::shared_ptr<Table> &split_points) {
  auto ctx = sample_result->GetContext();

  std::vector<std::shared_ptr<cylon::Table>> gather_results;
  // net::MPICommunicator comm;
  RETURN_CYLON_STATUS_IF_FAILED(
      ctx->GetCommunicator()->Gather(sample_result, 0, true, &gather_results));

  if (ctx->GetRank() == 0) {
    RETURN_CYLON_STATUS_IF_FAILED(
        DetermineSplitPoints(gather_results, sort_orders,
                             split_points, sample_result->GetContext()));
  }

  return ctx->GetCommunicator()->Bcast(&split_points, 0);
}

// return (index of) first element that is not less than the target element
int64_t tableBinarySearch(
    const std::shared_ptr<Table> &split_points,
    const std::shared_ptr<Table> &sorted_table,
    std::unique_ptr<DualTableRowIndexEqualTo>& equal_to,
    int64_t split_point_idx, int64_t l) {
  int64_t r = sorted_table->Rows() - 1;
  int L = l;

  while (r >= l) {
    int64_t m = (l + r) / 2;
    int compare_result_1 = equal_to->compare(m, util::SetBit(split_point_idx));
    int compare_result_2 =
        m == L ? -1 : equal_to->compare(m - 1, util::SetBit(split_point_idx));
    if (compare_result_1 >= 0 && compare_result_2 < 0)
      return m;
    else if (compare_result_1 < 0) {
      l = m + 1;
    } else {
      r = m - 1;
    }
  }

  return sorted_table->Rows();
}

Status GetSplitPointIndices(const std::shared_ptr<Table> &split_points,
                            const std::shared_ptr<Table> &sorted_table,
                            const std::vector<int> &sort_columns,
                            const std::vector<bool> &sort_order,
                            std::vector<uint32_t> &target_partition,
                            std::vector<uint32_t> &partition_hist) {
  // binary search
  int num_split_points = split_points->Rows();

  auto arrow_sorted_table = sorted_table->get_table();
  auto arrow_split_points = split_points->get_table();

  CYLON_ASSIGN_OR_RAISE(auto arrow_sorted_table_comb,
                        arrow_sorted_table->CombineChunks(
                            ToArrowPool(sorted_table->GetContext())));
  CYLON_ASSIGN_OR_RAISE(auto arrow_split_points_comb,
                        arrow_split_points->CombineChunks(
                            ToArrowPool(sorted_table->GetContext())));

  std::vector<int> split_points_sort_cols(split_points->Columns());
  std::iota(split_points_sort_cols.begin(), split_points_sort_cols.end(), 0);

  std::unique_ptr<DualTableRowIndexEqualTo> equal_to;
  RETURN_CYLON_STATUS_IF_FAILED(DualTableRowIndexEqualTo::Make(
      arrow_sorted_table_comb, arrow_split_points_comb, sort_columns,
      split_points_sort_cols, &equal_to, sort_order));

  int64_t num_rows = sorted_table->Rows();
  target_partition.resize(num_rows);
  partition_hist.resize(num_split_points + 1);
  int64_t l_idx = 0;

  for (int64_t i = 0; i < num_split_points; i++) {
    int64_t idx =
        tableBinarySearch(split_points, sorted_table, equal_to, i, l_idx);
    std::fill(target_partition.begin() + l_idx, target_partition.begin() + idx,
              i);
    partition_hist[i] = idx - l_idx;
    l_idx = idx;
  }

  std::fill(target_partition.begin() + l_idx, target_partition.end(),
            num_split_points);
  partition_hist[num_split_points] = num_rows - l_idx;

  return Status::OK();
}

/**
 * perform distributed sort on provided table
 * @param table
 * @param sort_columns sort based on these columns
 * @param sort_direction Sort direction 'true' indicates ascending ordering and false indicate descending ordering.
 * @param sorted_table resulting table
 * @return
 */
Status DistributedSortRegularSampling(const std::shared_ptr<Table> &table,
                                      const std::vector<int32_t> &sort_columns,
                                      const std::vector<bool> &sort_direction,
                                      std::shared_ptr<cylon::Table> &output,
                                      SortOptions sort_options) {
  if (sort_columns.size() > table->Columns()) {
    return Status(Code::ValueError,
                  "number of values in sort_column_indices can not larger than "
                  "the number of columns");
  } else if (sort_columns.size() != sort_direction.size()) {
    return Status(Code::ValueError,
                  "sizes of sort_column_indices and column_orders must match");
  }

  const auto &ctx = table->GetContext();
  int world_sz = ctx->GetWorldSize();

  if (world_sz == 1) {
    return Sort(table, sort_columns, output, sort_direction);
  }
  // locally sort
  std::shared_ptr<Table> local_sorted;
  RETURN_CYLON_STATUS_IF_FAILED(
      Sort(table, sort_columns, local_sorted, sort_direction));

  const int SAMPLING_RATIO =
      sort_options.num_samples == 0 ? 2 : sort_options.num_samples;

  std::vector<uint32_t> target_partitions, partition_hist;
  std::vector<std::shared_ptr<arrow::Table>> split_tables;

  // sample the sorted table with sort columns and create a table
  int sample_count = ctx->GetWorldSize() * SAMPLING_RATIO;
  sample_count = std::min((int64_t)sample_count, table->Rows());

  // sample_result only contains sorted columns
  std::shared_ptr<Table> sample_result;

  RETURN_CYLON_STATUS_IF_FAILED(
      SampleTableUniform(local_sorted, sample_count, sort_columns, sample_result, ctx));

  // determine split point, split_points only contains sorted columns
  std::shared_ptr<Table> split_points;
  RETURN_CYLON_STATUS_IF_FAILED(GetSplitPoints(
      sample_result, sort_direction, world_sz - 1, split_points));

  // construct target_partition, partition_hist
  RETURN_CYLON_STATUS_IF_FAILED(
      GetSplitPointIndices(split_points, local_sorted, sort_columns,
                           sort_direction, target_partitions, partition_hist));

  // split and all_to_all
  RETURN_CYLON_STATUS_IF_FAILED(Split(local_sorted, world_sz, target_partitions,
                                      partition_hist, split_tables));

  // we are going to free if retain is set to false. therefore, we need to make
  // a copy of schema
  std::shared_ptr<arrow::Schema> schema = table->get_table()->schema();
  //   if (!table->IsRetain()) {
  //     const_cast<std::shared_ptr<Table> &>(table).reset();
  //   }
  std::vector<std::shared_ptr<Table>> all_to_all_result;
  RETURN_CYLON_STATUS_IF_FAILED(all_to_all_arrow_tables_separated_cylon_table(
      ctx, schema, split_tables, all_to_all_result));

  return MergeSortedTable(all_to_all_result, sort_columns, sort_direction, output, 
    sort_options.sort_method == SortOptions::REGULAR_SAMPLE_MERGE);
}

Status DistributedSortInitialSampling(const std::shared_ptr<Table> &table,
                       const std::vector<int> &sort_columns,
                       std::shared_ptr<Table> &output,
                       const std::vector<bool> &sort_direction,
                       SortOptions sort_options) {
  const auto &ctx = table->GetContext();
  int world_sz = ctx->GetWorldSize();

  std::shared_ptr<arrow::Table> arrow_table, sorted_table;
  // first do distributed sort partitioning
  if (world_sz == 1) {
    arrow_table = table->get_table();
  } else {
    std::vector<uint32_t> target_partitions, partition_hist;
    std::vector<std::shared_ptr<arrow::Table>> split_tables;

    RETURN_CYLON_STATUS_IF_FAILED(MapToSortPartitions(table,
                                                      sort_columns[0],
                                                      world_sz,
                                                      target_partitions,
                                                      partition_hist,
                                                      sort_direction[0],
                                                      sort_options.num_samples,
                                                      sort_options.num_bins));

    RETURN_CYLON_STATUS_IF_FAILED(Split(table,
                                        world_sz,
                                        target_partitions,
                                        partition_hist,
                                        split_tables));

    // we are going to free if retain is set to false. therefore, we need to make a copy of schema
    std::shared_ptr<arrow::Schema> schema = table->get_table()->schema();
    if (!table->IsRetain()) {
      const_cast<std::shared_ptr<Table> &>(table).reset();
    }

    RETURN_CYLON_STATUS_IF_FAILED(all_to_all_arrow_tables(ctx, schema, split_tables, arrow_table));
  }

  // then do a local sort
  if (sort_columns.size() == 1) {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::SortTable(arrow_table,
                                                        sort_columns[0],
                                                        ToArrowPool(ctx),
                                                        sorted_table, sort_direction[0]));
  } else {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::SortTableMultiColumns(arrow_table,
                                                                    sort_columns,
                                                                    ToArrowPool(ctx),
                                                                    sorted_table,
                                                                    sort_direction));
  }

  return Table::FromArrowTable(ctx, sorted_table, output);
}

Status DistributedSort(const std::shared_ptr<Table> &table,
                       int sort_column,
                       std::shared_ptr<Table> &output,
                       bool ascending,
                       SortOptions sort_options) {
  return DistributedSort(table, std::vector<int>{sort_column}, output, std::vector<bool>{ascending},
                         sort_options);
}

Status DistributedSort(const std::shared_ptr<Table> &table,
                       const std::vector<int> &sort_columns,
                       std::shared_ptr<Table> &output,
                       const std::vector<bool> &sort_direction,
                       SortOptions sort_options) {
  if(sort_options.sort_method == sort_options.INITIAL_SAMPLE) {
    return DistributedSortInitialSampling(table, sort_columns, output, sort_direction, sort_options);
  } else {
    return DistributedSortRegularSampling(table, sort_columns, sort_direction, output, sort_options);
  }
}

Status HashPartition(const std::shared_ptr<Table> &table, const std::vector<int> &hash_columns,
                     int no_of_partitions,
                     std::unordered_map<int, std::shared_ptr<cylon::Table>> *out) {
  // keep arrays for each target, these arrays are used for creating the table
  std::vector<uint32_t> outPartitions, counts;
  RETURN_CYLON_STATUS_IF_FAILED(MapToHashPartitions(table,
                                                    hash_columns,
                                                    no_of_partitions,
                                                    outPartitions,
                                                    counts));

  std::vector<std::shared_ptr<arrow::Table>> partitioned_tables;
  RETURN_CYLON_STATUS_IF_FAILED(Split(table,
                                      no_of_partitions,
                                      outPartitions,
                                      counts,
                                      partitioned_tables));

  const auto &ctx = table->GetContext();
  out->reserve(no_of_partitions);
  for (int i = 0; i < no_of_partitions; i++) {
    out->emplace(i, std::make_shared<Table>(ctx, partitioned_tables[i]));
  }

  return Status::OK();
}

arrow::Status create_table_with_duplicate_index(arrow::MemoryPool *pool,
                                                std::shared_ptr<arrow::Table> &table,
                                                size_t index_column) {
  const std::vector<std::shared_ptr<arrow::ChunkedArray>> &chunk_arrays = table->columns();
  std::vector<std::shared_ptr<arrow::ChunkedArray>> new_arrays;
  new_arrays.reserve(chunk_arrays.size());
  for (size_t i = 0; i < chunk_arrays.size(); i++) {
    if (i != index_column) {
      new_arrays.push_back(chunk_arrays[i]);
    } else {
      std::shared_ptr<arrow::ChunkedArray> new_c_array;
      RETURN_ARROW_STATUS_IF_FAILED(cylon::util::Duplicate(chunk_arrays[i], pool, new_c_array));
      new_arrays.emplace_back(std::move(new_c_array));
    }
  }
  table = arrow::Table::Make(table->schema(), std::move(new_arrays));
  return arrow::Status::OK();
}

Status Join(const std::shared_ptr<Table> &left, const std::shared_ptr<Table> &right,
            const join::config::JoinConfig &join_config, std::shared_ptr<cylon::Table> &out) {
  if (left == NULLPTR) {
    return Status(Code::KeyError, "Couldn't find the left table");
  } else if (right == NULLPTR) {
    return Status(Code::KeyError, "Couldn't find the right table");
  } else {
    std::shared_ptr<arrow::Table> table, left_table, right_table;
    const auto &ctx = left->GetContext();
    auto pool = cylon::ToArrowPool(ctx);

    left->ToArrowTable(left_table);
    right->ToArrowTable(right_table);
    // if it is a sort algorithm and certain key types, we are going to do an in-place sort
    if (!join_config.IsMultiColumn() && join_config.GetAlgorithm() == cylon::join::config::SORT) {
      int lIndex = join_config.GetLeftColumnIdx()[0];
      int rIndex = join_config.GetRightColumnIdx()[0];
      auto left_type = left_table->column(lIndex)->type()->id();
      if (cylon::join::util::is_inplace_join_possible(left_type)) {
        // we don't have to copy if the table is freed
        if (left->IsRetain()) {
          RETURN_CYLON_STATUS_IF_ARROW_FAILED(
              create_table_with_duplicate_index(pool, left_table, lIndex));
        }
        if (right->IsRetain()) {
          RETURN_CYLON_STATUS_IF_ARROW_FAILED(
              create_table_with_duplicate_index(pool, right_table, rIndex));
        }
      }
    }

    RETURN_CYLON_STATUS_IF_FAILED(join::JoinTables(left_table, right_table, join_config,
                                                   &table, pool));
    return Table::FromArrowTable(ctx, std::move(table), out);
  }
}

Status Table::ToArrowTable(std::shared_ptr<arrow::Table> &out) {
  out = table_;
  return Status::OK();
}

Status DistributedJoin(const std::shared_ptr<Table> &left, const std::shared_ptr<Table> &right,
                       const join::config::JoinConfig &join_config,
                       std::shared_ptr<cylon::Table> &out) {
  // check whether the world size is 1
  const auto &ctx = left->GetContext();
  if (ctx->GetWorldSize() == 1) {
    return Join(left, right, join_config, out);
  }

  std::shared_ptr<arrow::Table> left_final_table, right_final_table;
  RETURN_CYLON_STATUS_IF_FAILED(shuffle_two_tables_by_hashing(ctx,
                                                              left,
                                                              join_config.GetLeftColumnIdx(),
                                                              right,
                                                              join_config.GetRightColumnIdx(),
                                                              left_final_table,
                                                              right_final_table));

  std::shared_ptr<arrow::Table> table;
  RETURN_CYLON_STATUS_IF_FAILED(join::JoinTables(left_final_table, right_final_table,
                                                 join_config, &table, cylon::ToArrowPool(ctx)));
  return Table::FromArrowTable(ctx, std::move(table), out);
}

Status Select(const std::shared_ptr<Table> &table, const std::function<bool(cylon::Row)> &selector,
              std::shared_ptr<Table> &out) {
  // boolean builder to hold the mask
  const auto &ctx = table->GetContext();
  const auto &table_ = table->get_table();
  auto row = cylon::Row(table_);
  auto pool = cylon::ToArrowPool(ctx);
  std::shared_ptr<arrow::Table> out_table;

  auto kI = table->Rows();
  if (kI) {
    arrow::BooleanBuilder boolean_builder(pool);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(boolean_builder.Reserve(kI));

    for (int64_t row_index = 0; row_index < kI; row_index++) {
      row.SetIndex(row_index);
      boolean_builder.UnsafeAppend(selector(row));
    }
    // building the mask
    std::shared_ptr<arrow::Array> mask;
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(boolean_builder.Finish(&mask));

    const arrow::Result<arrow::Datum> &filter_res = arrow::compute::Filter(table_, mask);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(filter_res.status());

    out_table = filter_res.ValueOrDie().table();
  } else {
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(util::Duplicate(table_, pool, out_table));
  }
  out = std::make_shared<cylon::Table>(ctx, out_table);
  return Status::OK();
}

Status Union(const std::shared_ptr<Table> &first, const std::shared_ptr<Table> &second,
             std::shared_ptr<Table> &out) {
  std::shared_ptr<arrow::Table> ltab = first->get_table();
  std::shared_ptr<arrow::Table> rtab = second->get_table();
  const auto &ctx = first->GetContext();
  auto pool = ToArrowPool(ctx);

  COMBINE_CHUNKS_RETURN_CYLON_STATUS(ltab, pool);
  COMBINE_CHUNKS_RETURN_CYLON_STATUS(rtab, pool);

  RETURN_CYLON_STATUS_IF_FAILED(VerifyTableSchema(ltab, rtab));

  std::unique_ptr<DualTableRowIndexHash> hash;
  RETURN_CYLON_STATUS_IF_FAILED(DualTableRowIndexHash::Make(ltab, rtab, &hash));

  std::unique_ptr<DualTableRowIndexEqualTo> equal_to;
  RETURN_CYLON_STATUS_IF_FAILED(DualTableRowIndexEqualTo::Make(ltab, rtab, &equal_to));

  const auto buckets_pre_alloc = (ltab->num_rows() + rtab->num_rows());
  ska::bytell_hash_set<int64_t, DualTableRowIndexHash, DualTableRowIndexEqualTo>
      rows_set(buckets_pre_alloc, *hash, *equal_to);

  arrow::compute::ExecContext exec_context(pool);

  std::shared_ptr<arrow::Array> mask;

  // insert first table to the row set
  arrow::BooleanBuilder mask_builder(pool);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(mask_builder.Reserve(ltab->num_rows()));
  for (int64_t i = 0; i < ltab->num_rows(); i++) {
    const auto &res = rows_set.insert(i);
    // if res.second == true: it is a unique value
    // else: its already available
    mask_builder.UnsafeAppend(res.second);
  }
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(mask_builder.Finish(&mask));

  const auto &options = arrow::compute::FilterOptions::Defaults();
  CYLON_ASSIGN_OR_RAISE(
      auto l_res, arrow::compute::Filter(ltab, mask, options, &exec_context));
  // filtered first table
  const std::shared_ptr<arrow::Table> &f_ltab = l_res.table();

  // insert second table to the row set
  mask_builder.Reset();
  mask.reset();
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(mask_builder.Reserve(rtab->num_rows()));
  for (int64_t i = 0; i < rtab->num_rows(); i++) {
    // setting the leading bit to 1 since we are inserting the second table
    const auto &res = rows_set.insert(util::SetBit(i));
    // if res.second == true: it is a unique value
    // else: its already available
    mask_builder.UnsafeAppend(res.second);
  }
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(mask_builder.Finish(&mask));

  CYLON_ASSIGN_OR_RAISE(
      auto r_res, arrow::compute::Filter(rtab, mask, options, &exec_context))
  // filtered second table
  const std::shared_ptr<arrow::Table> &f_rtab = r_res.table();

  // concat filtered tables
  CYLON_ASSIGN_OR_RAISE(auto concat,
                        arrow::ConcatenateTables({f_ltab, f_rtab},
                                                 arrow::ConcatenateTablesOptions::Defaults(),
                                                 pool))
  // combine chunks
  CYLON_ASSIGN_OR_RAISE(auto merge, concat->CombineChunks())

  return Table::FromArrowTable(ctx, std::move(merge), out);
}

Status Subtract(const std::shared_ptr<Table> &first, const std::shared_ptr<Table> &second,
                std::shared_ptr<Table> &out) {
  std::shared_ptr<arrow::Table> ltab = first->get_table();
  std::shared_ptr<arrow::Table> rtab = second->get_table();
  const auto &ctx = first->GetContext();
  auto pool = ToArrowPool(ctx);

  COMBINE_CHUNKS_RETURN_CYLON_STATUS(ltab, pool);
  COMBINE_CHUNKS_RETURN_CYLON_STATUS(rtab, pool);

  RETURN_CYLON_STATUS_IF_FAILED(VerifyTableSchema(ltab, rtab));

  std::unique_ptr<DualTableRowIndexHash> hash;
  RETURN_CYLON_STATUS_IF_FAILED(DualTableRowIndexHash::Make(ltab, rtab, &hash));

  std::unique_ptr<DualTableRowIndexEqualTo> equal_to;
  RETURN_CYLON_STATUS_IF_FAILED(DualTableRowIndexEqualTo::Make(ltab, rtab, &equal_to));

  const auto buckets_pre_alloc = ltab->num_rows();
  ska::bytell_hash_set<int64_t, DualTableRowIndexHash, DualTableRowIndexEqualTo>
      rows_set(buckets_pre_alloc, *hash, *equal_to);

  arrow::compute::ExecContext exec_context(pool);

  // create a bitmask
  arrow::BooleanBuilder builder;
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Reserve(ltab->num_rows()));

  // insert left table to row_set
  for (int64_t i = 0; i < ltab->num_rows(); i++) {
    const auto &res = rows_set.insert(i);
    builder.UnsafeAppend(res.second);
  }

  // now create the mask
  CYLON_ASSIGN_OR_RAISE(auto mask, builder.Finish())
  uint8_t *bit_buf = mask->data()->buffers[1]->mutable_data();

  // let's probe right rows against the rows set
  for (int64_t i = 0; i < rtab->num_rows(); i++) {
    // setting the leading bit to 1 since we are inserting the second table
    const auto &res = rows_set.find(util::SetBit(i));
    if (res != rows_set.end()) { // clear bit if we find matches while probing
      arrow::BitUtil::ClearBit(bit_buf, *res);
    }
  }

  // filtered first table
  CYLON_ASSIGN_OR_RAISE(auto l_res, arrow::compute::Filter(ltab, mask,
                                                           arrow::compute::FilterOptions::Defaults(),
                                                           &exec_context))
  return Table::FromArrowTable(ctx, l_res.table(), out);
}

Status Intersect(const std::shared_ptr<Table> &first,
                 const std::shared_ptr<Table> &second,
                 std::shared_ptr<Table> &out) {
  std::shared_ptr<arrow::Table> ltab = first->get_table();
  std::shared_ptr<arrow::Table> rtab = second->get_table();

  const auto &ctx = first->GetContext();
  auto pool = ToArrowPool(ctx);

  COMBINE_CHUNKS_RETURN_CYLON_STATUS(ltab, pool);
  COMBINE_CHUNKS_RETURN_CYLON_STATUS(rtab, pool);

  RETURN_CYLON_STATUS_IF_FAILED(VerifyTableSchema(ltab, rtab));

  std::unique_ptr<DualTableRowIndexHash> hash;
  RETURN_CYLON_STATUS_IF_FAILED(DualTableRowIndexHash::Make(ltab, rtab, &hash));

  std::unique_ptr<DualTableRowIndexEqualTo> equal_to;
  RETURN_CYLON_STATUS_IF_FAILED(DualTableRowIndexEqualTo::Make(ltab, rtab, &equal_to));

  const auto buckets_pre_alloc = ltab->num_rows();
  ska::bytell_hash_set<int64_t, DualTableRowIndexHash, DualTableRowIndexEqualTo>
      rows_set(buckets_pre_alloc, *hash, *equal_to);

  arrow::compute::ExecContext exec_context(pool);

  // insert left table to row_set
  for (int64_t i = 0; i < ltab->num_rows(); i++) {
    rows_set.insert(i);
  }

  // create a bitmask
  std::vector<bool> bitmask(ltab->num_rows(), false);

  // let's probe right rows against the rows set
  for (int64_t i = 0; i < rtab->num_rows(); i++) {
    // setting the leading bit to 1 since we are inserting the second table
    const auto &res = rows_set.find(util::SetBit(i));
    if (res != rows_set.end()) {
      bitmask[*res] = true;
    }
  }

  // convert vector<bool> to BooleanArray
  arrow::BooleanBuilder builder;
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.AppendValues(bitmask));
  std::shared_ptr<arrow::Array> mask;
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(builder.Finish(&mask));

  const arrow::Result<arrow::Datum> &l_res = arrow::compute::Filter(ltab,
                                                                    mask,
                                                                    arrow::compute::FilterOptions::Defaults(),
                                                                    &exec_context);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(l_res.status());

  // filtered second table
  std::shared_ptr<arrow::Table> intersect_tab = l_res.ValueOrDie().table();

  out = std::make_shared<cylon::Table>(ctx, intersect_tab);
  return Status::OK();
}

//typedef Status (*LocalSetOperation)(const std::shared_ptr<cylon::Table> &,
//									const std::shared_ptr<cylon::Table> &,
//									std::shared_ptr<cylon::Table> &);

template<typename LocalSetOperation>
static inline Status do_dist_set_op(LocalSetOperation local_operation,
                                    const std::shared_ptr<Table> &table_left,
                                    const std::shared_ptr<Table> &table_right,
                                    std::shared_ptr<cylon::Table> &out) {
  // extract the tables out
  auto left = table_left->get_table();
  auto right = table_right->get_table();
  const auto &ctx = table_left->GetContext();

  RETURN_CYLON_STATUS_IF_FAILED(VerifyTableSchema(left, right));

  if (ctx->GetWorldSize() == 1) {
    return local_operation(table_left, table_right, out);
  }

  std::vector<int32_t> hash_columns;
  hash_columns.reserve(left->num_columns());
  for (int kI = 0; kI < left->num_columns(); ++kI) {
    hash_columns.push_back(kI);
  }

  std::shared_ptr<arrow::Table> left_final_table;
  std::shared_ptr<arrow::Table> right_final_table;
  RETURN_CYLON_STATUS_IF_FAILED(
      shuffle_two_tables_by_hashing(ctx, table_left, hash_columns, table_right, hash_columns,
                                    left_final_table, right_final_table));

  std::shared_ptr<cylon::Table> left_tab = std::make_shared<cylon::Table>(ctx, left_final_table);
  std::shared_ptr<cylon::Table> right_tab = std::make_shared<cylon::Table>(ctx, right_final_table);
  // now do the local union
  std::shared_ptr<arrow::Table> table;
  return local_operation(left_tab, right_tab, out);
}

Status DistributedUnion(const std::shared_ptr<Table> &left, const std::shared_ptr<Table> &right,
                        std::shared_ptr<Table> &out) {
  return do_dist_set_op(&Union, left, right, out);
}

Status DistributedSubtract(const std::shared_ptr<Table> &left, const std::shared_ptr<Table> &right,
                           std::shared_ptr<Table> &out) {
  return do_dist_set_op(&Subtract, left, right, out);
}

Status DistributedIntersect(const std::shared_ptr<Table> &left, const std::shared_ptr<Table> &right,
                            std::shared_ptr<Table> &out) {
  return do_dist_set_op(&Intersect, left, right, out);
}

void ReadCSVThread(const std::shared_ptr<CylonContext> &ctx, const std::string &path,
                   std::shared_ptr<cylon::Table> *table,
                   const cylon::io::config::CSVReadOptions &options,
                   const std::shared_ptr<std::promise<Status>> &status_promise) {
//  const std::shared_ptr<CylonContext> &ctx_ = ctx;  // make a copy of the shared ptr
  status_promise->set_value(FromCSV(ctx, path, *table, options));
}

Status FromCSV(const std::shared_ptr<CylonContext> &ctx, const std::vector<std::string> &paths,
               const std::vector<std::shared_ptr<Table> *> &tableOuts,
               const io::config::CSVReadOptions &options) {
  if (options.IsConcurrentFileReads()) {
    std::vector<std::pair<std::future<Status>, std::thread>> futures;
    futures.reserve(paths.size());
    for (uint64_t kI = 0; kI < paths.size(); ++kI) {
      auto read_promise = std::make_shared<std::promise<Status>>();
      //	  auto context = ctx.get();
      futures.emplace_back(
          read_promise->get_future(),
          std::thread(ReadCSVThread,
                      std::cref(ctx),
                      std::cref(paths[kI]),
                      tableOuts[kI],
                      std::cref(options),
                      read_promise));
    }
    bool all_passed = true;
    for (auto &future: futures) {
      auto status = future.first.get();
      all_passed &= status.is_ok();
      future.second.join();
    }
    return all_passed ? Status::OK() : Status(cylon::IOError, "Failed to read the csv files");
  } else {
    auto status = Status::OK();
    for (std::size_t kI = 0; kI < paths.size(); ++kI) {
      status = FromCSV(ctx, paths[kI], *tableOuts[kI], options);
      if (!status.is_ok()) {
        return status;
      }
    }
    return status;
  }
}

Status Project(const std::shared_ptr<Table> &table, const std::vector<int32_t> &project_columns,
               std::shared_ptr<Table> &out) {
  std::vector<std::shared_ptr<arrow::Field>> schema_vector;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> column_arrays;
  schema_vector.reserve(project_columns.size());
  column_arrays.reserve(project_columns.size());

  auto table_ = table->get_table();
  const auto &ctx = table->GetContext();

  for (auto const &col_index: project_columns) {
    schema_vector.push_back(table_->field(col_index));
    column_arrays.push_back(table_->column(col_index));
  }

  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  std::shared_ptr<arrow::Table> ar_table = arrow::Table::Make(schema, column_arrays);
  out = std::make_shared<cylon::Table>(ctx, ar_table);
  return Status::OK();
}

Status Table::PrintToOStream(std::ostream &out) const {
  return PrintToOStream(0, Columns(), 0, Rows(), out);
}

Status Table::PrintToOStream(int col1, int col2, int64_t row1, int64_t row2, std::ostream &out,
                             char delimiter, bool use_custom_header,
                             const std::vector<std::string> &headers) const {
  if (table_ != NULLPTR) {
    // print the headers
    if (use_custom_header) {
      // check if the headers are valid
      if (headers.size() != (size_t) table_->num_columns()) {
        return Status(
            cylon::Code::IndexError,
            "Provided headers doesn't match with the number of columns of the table. Given " +
                std::to_string(headers.size()) + ", Expected " +
                std::to_string(table_->num_columns()));
      }

      for (int col = col1; col < col2; col++) {
        out << headers[col];
        if (col != col2 - 1) {
          out << delimiter;
        } else {
          out << std::endl;
        }
      }
    } else {
      const auto &field_names = table_->schema()->field_names();
      for (int col = col1; col < col2; col++) {
        out << field_names[col];
        if (col != col2 - 1) {
          out << delimiter;
        } else {
          out << std::endl;
        }
      }
    }
    for (int row = row1; row < row2; row++) {
      for (int col = col1; col < col2; col++) {
        auto column = table_->column(col);
        int rowCount = 0;
        for (int chunk = 0; chunk < column->num_chunks(); chunk++) {
          auto array = column->chunk(chunk);
          if (rowCount <= row && rowCount + array->length() > row) {
            // print this array
            out << cylon::util::array_to_string(array, row - rowCount);
            if (col != col2 - 1) {
              out << delimiter;
            }
            break;
          }
          rowCount += array->length();
        }
      }
      out << std::endl;
    }
  }
  return Status(Code::OK);
}

const std::shared_ptr<arrow::Table> &Table::get_table() const { return table_; }

bool Table::IsRetain() const { return retain_; }

Status Shuffle(const std::shared_ptr<Table> &table, const std::vector<int> &hash_columns,
               std::shared_ptr<cylon::Table> &output) {
  const auto &ctx_ = table->GetContext();
  std::shared_ptr<arrow::Table> table_out;
  RETURN_CYLON_STATUS_IF_FAILED(shuffle_table_by_hashing(ctx_, table, hash_columns, table_out));
  return cylon::Table::FromArrowTable(ctx_, std::move(table_out), output);
}

Status Unique(const std::shared_ptr<Table> &in, const std::vector<int> &cols,
              std::shared_ptr<cylon::Table> &out, bool first) {
#ifdef CYLON_DEBUG
  auto p1 = std::chrono::high_resolution_clock::now();
#endif
  const auto &ctx = in->GetContext();
  auto pool = cylon::ToArrowPool(ctx);
  std::shared_ptr<arrow::Table> out_table, in_table = in->get_table();

  if (!in->Empty()) {
    if (in_table->column(0)->num_chunks() > 1) {
      CYLON_ASSIGN_OR_RAISE(in_table, in_table->CombineChunks(pool))
    }

    std::unique_ptr<TableRowIndexEqualTo> row_comp;
    RETURN_CYLON_STATUS_IF_FAILED(TableRowIndexEqualTo::Make(in_table, cols, &row_comp));

    std::unique_ptr<TableRowIndexHash> row_hash;
    RETURN_CYLON_STATUS_IF_FAILED(TableRowIndexHash::Make(in_table, cols, &row_hash));

    const int64_t num_rows = in_table->num_rows();
    ska::bytell_hash_set<int64_t, TableRowIndexHash, TableRowIndexEqualTo>
        rows_set(num_rows, *row_hash, *row_comp);

    arrow::Int64Builder filter(pool);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(filter.Reserve(num_rows));
#ifdef CYLON_DEBUG
    auto p2 = std::chrono::high_resolution_clock::now();
#endif
    if (first) {
      for (int64_t row = 0; row < num_rows; ++row) {
        const auto &res = rows_set.insert(row);
        if (res.second) {
          filter.UnsafeAppend(row);
        }
      }
    } else {
      for (int64_t row = num_rows - 1; row > 0; --row) {
        const auto &res = rows_set.insert(row);
        if (res.second) {
          filter.UnsafeAppend(row);
        }
      }
    }
#ifdef CYLON_DEBUG
    auto p3 = std::chrono::high_resolution_clock::now();
#endif
    rows_set.clear();
#ifdef CYLON_DEBUG
    auto p4 = std::chrono::high_resolution_clock::now();
#endif
    CYLON_ASSIGN_OR_RAISE(auto take_arr, filter.Finish());
    CYLON_ASSIGN_OR_RAISE(auto take_res, arrow::compute::Take(in_table, take_arr))
    out_table = take_res.table();

#ifdef CYLON_DEBUG
    auto p5 = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "P1 " << std::chrono::duration_cast<std::chrono::milliseconds>(p2 - p1).count()
              << " P2 " << std::chrono::duration_cast<std::chrono::milliseconds>(p3 - p2).count()
              << " P3 " << std::chrono::duration_cast<std::chrono::milliseconds>(p4 - p3).count()
              << " P4 " << std::chrono::duration_cast<std::chrono::milliseconds>(p5 - p4).count()
              << " tot " << std::chrono::duration_cast<std::chrono::milliseconds>(p5 - p1).count()
              << " tot " << rows_set.load_factor() << " " << rows_set.bucket_count();
#endif
  } else {
    out_table = in_table;
  }
  return Table::FromArrowTable(ctx, std::move(out_table), out);
}

Status DistributedUnique(const std::shared_ptr<Table> &in, const std::vector<int> &cols,
                         std::shared_ptr<cylon::Table> &out) {
  const auto &ctx = in->GetContext();
  if (ctx->GetWorldSize() == 1) {
    return Unique(in, cols, out);
  }

  std::shared_ptr<cylon::Table> shuffle_out;
  RETURN_CYLON_STATUS_IF_FAILED(cylon::Shuffle(in, cols, shuffle_out));

  return Unique(shuffle_out, cols, out);
}

Status Equals(const std::shared_ptr<cylon::Table> &a, const std::shared_ptr<cylon::Table> &b,
              bool &result, bool ordered) {
  if (ordered) {
    result = a->get_table()->Equals(*b->get_table());
  } else {
    result = false;
    if (a->Columns() != b->Columns()) {
      return Status::OK();
    }
    RETURN_CYLON_STATUS_IF_FAILED(VerifyTableSchema(a->get_table(), b->get_table()));

    int col = a->Columns();

    std::vector<int32_t> indices(col);
    std::iota(indices.begin(), indices.end(), 0);

    std::shared_ptr<cylon::Table> out_a, out_b;
    RETURN_CYLON_STATUS_IF_FAILED(Sort(a, indices, out_a, true));
    RETURN_CYLON_STATUS_IF_FAILED(Sort(b, indices, out_b, true));

    result = out_a->get_table()->Equals(*out_b->get_table());
  }
  return Status::OK();
}

static Status RepartitionToMatchOtherTable(const std::shared_ptr<cylon::Table> &a,
                                           const std::shared_ptr<cylon::Table> &b,
                                           std::shared_ptr<cylon::Table> *b_out) {
  int world_size = a->GetContext()->GetWorldSize();
  int64_t num_row = a->Rows();

  if (num_row == 0) {
    *b_out = a;
    return Status::OK();
  }

  std::vector<int64_t> rows_per_partition;
  RETURN_CYLON_STATUS_IF_FAILED(mpi::AllGather(a->GetContext(),
                                               {num_row},
                                               world_size,
                                               rows_per_partition));

  return Repartition(b, rows_per_partition, b_out);
}

Status DistributedEquals(const std::shared_ptr<cylon::Table> &a,
                         const std::shared_ptr<cylon::Table> &b,
                         bool &result,
                         bool ordered) {
  bool subResult;
  RETURN_CYLON_STATUS_IF_FAILED(VerifyTableSchema(a->get_table(), b->get_table()));

  if (!ordered) {
    int col = a->Columns();
    std::vector<int32_t> indices(col);
    std::vector<bool> column_orders(col, true);
    std::iota(indices.begin(), indices.end(), 0);

    std::shared_ptr<cylon::Table> a_sorted, b_sorted;
    RETURN_CYLON_STATUS_IF_FAILED(DistributedSort(a, indices, a_sorted, column_orders));
    RETURN_CYLON_STATUS_IF_FAILED(DistributedSort(b, indices, b_sorted, column_orders));

    std::shared_ptr<cylon::Table> b_repartitioned;
    RETURN_CYLON_STATUS_IF_FAILED(RepartitionToMatchOtherTable(a_sorted, b_sorted, &b_repartitioned));

    RETURN_CYLON_STATUS_IF_FAILED(Equals(a_sorted, b_repartitioned, subResult));
  } else {
    std::shared_ptr<cylon::Table> b_repartitioned;
    RETURN_CYLON_STATUS_IF_FAILED(RepartitionToMatchOtherTable(a, b, &b_repartitioned));
    RETURN_CYLON_STATUS_IF_FAILED(Equals(a, b_repartitioned, subResult, true));
  }

  RETURN_CYLON_STATUS_IF_FAILED(mpi::AllReduce(a->GetContext(),
                                               &subResult,
                                               &result,
                                               1,
                                               Bool(),
                                               cylon::net::LAND));
  return Status::OK();
}

Status Repartition(const std::shared_ptr<cylon::Table> &table,
                   const std::vector<int64_t> &rows_per_partition,
                   const std::vector<int> &receive_build_rank_order,
                   std::shared_ptr<Table> *output) {
  int world_size = table->GetContext()->GetWorldSize();
  int rank = table->GetContext()->GetRank();
  int num_row = (int) table->Rows();

  if (rows_per_partition.size() != (size_t) world_size) {
    return Status(cylon::Code::ValueError,
                  "rows_per_partition size does not align with world size. Received " +
                      std::to_string(rows_per_partition.size()) + ", Expected " +
                      std::to_string(world_size));
  }

  std::vector<int64_t> sizes;
  RETURN_CYLON_STATUS_IF_FAILED(mpi::AllGather(table->GetContext(), {num_row}, world_size, sizes));

  auto out_partitions_temp = RowIndicesToAll(rank, sizes, rows_per_partition);
  uint32_t no_of_partitions = receive_build_rank_order.size();
  std::vector<uint32_t> out_partitions(num_row);
  auto begin = out_partitions.begin();

  for (int i = 0; i < (int) receive_build_rank_order.size(); i++) {
    int64_t s = out_partitions_temp[i], t = out_partitions_temp[i + 1];
    fill(begin + s, begin + t, receive_build_rank_order[i]);
  }

  std::vector<std::shared_ptr<arrow::Table>> partitioned_tables;
  RETURN_CYLON_STATUS_IF_FAILED(Split(table, no_of_partitions, out_partitions, partitioned_tables));

  const auto &schema = table->get_table()->schema();

  if (!table->IsRetain()) {
    const_cast<std::shared_ptr<Table> &>(table).reset();
  }

  // all_to_all, but preserves relative rank order
  std::shared_ptr<arrow::Table> table_out;
  RETURN_CYLON_STATUS_IF_FAILED(all_to_all_arrow_tables_preserve_order(table->GetContext(),
                                                                       schema,
                                                                       partitioned_tables,
                                                                       table_out));

  return Table::FromArrowTable(table->GetContext(), std::move(table_out), *output);
}

Status Repartition(const std::shared_ptr<cylon::Table> &table,
                   const std::vector<int64_t> &rows_per_partition,
                   std::shared_ptr<cylon::Table> *output) {
  // should be refactored after mpi_send and mpi_receive are implemented
  int world_size = table->GetContext()->GetWorldSize();
  std::vector<int32_t> indices(world_size);
  std::iota(indices.begin(), indices.end(), 0);

  return Repartition(table, rows_per_partition, indices, output);
}

// repartition to `world_size` number of partitions evenly
Status Repartition(const std::shared_ptr<cylon::Table> &table,
                   std::shared_ptr<cylon::Table> *output) {
  int world_size = table->GetContext()->GetWorldSize();
  std::vector<int64_t> size = {table->Rows()};
  std::vector<int64_t> sizes;
  RETURN_CYLON_STATUS_IF_FAILED(mpi::AllGather(table->GetContext(), size, world_size, sizes));

  auto result = DivideRowsEvenly(sizes);
  return Repartition(table, result, output);
}

std::shared_ptr<BaseArrowIndex> Table::GetArrowIndex() { return base_arrow_index_; }

Status Table::SetArrowIndex(std::shared_ptr<cylon::BaseArrowIndex> &index, bool drop_index) {
  if (table_->column(0)->num_chunks() > 1) {
    const arrow::Result<std::shared_ptr<arrow::Table>> &res =
        table_->CombineChunks(cylon::ToArrowPool(ctx));
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status());
    table_ = res.ValueOrDie();
  }

  base_arrow_index_ = index;

  if (drop_index) {
    arrow::Result<std::shared_ptr<arrow::Table>> result =
        table_->RemoveColumn(base_arrow_index_->GetColId());
    if (result.status() != arrow::Status::OK()) {
      LOG(ERROR) << "Column removal failed ";
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
    }
    table_ = std::move(result.ValueOrDie());
  }

  return Status::OK();
}

Status Table::ResetArrowIndex(bool drop) {
  if (base_arrow_index_) {
    if (typeid(base_arrow_index_) == typeid(cylon::ArrowRangeIndex)) {
      LOG(INFO) << "Table contains a range index";
    } else {
      LOG(INFO) << "Table contains a non-range index";
      auto index_arr = base_arrow_index_->GetIndexArray();
      auto pool = cylon::ToArrowPool(ctx);
      base_arrow_index_ = std::make_shared<cylon::ArrowRangeIndex>(0, table_->num_rows(), 1, pool);
      if (!drop) {
        LOG(INFO) << "Reset Index Drop case";
        AddColumn(0, "index", index_arr);
      }
    }
  }
  return Status::OK();
}

Status Table::AddColumn(int32_t position, const std::string &column_name,
                        std::shared_ptr<arrow::Array> input_column) {
  if (input_column->length() != table_->num_rows()) {
    return {cylon::Code::CapacityError,
            "New column length must match the number of rows in the table"};
  }
  auto field = arrow::field(column_name, input_column->type());
  auto chunked_array = std::make_shared<arrow::ChunkedArray>(std::move(input_column));
  CYLON_ASSIGN_OR_RAISE(table_,
                        table_->AddColumn(position, std::move(field), std::move(chunked_array)))
  return Status::OK();
}

const std::shared_ptr<cylon::CylonContext> &Table::GetContext() const {
  return ctx;
}

Table::Table(const std::shared_ptr<CylonContext> &ctx, std::shared_ptr<arrow::Table> tab)
    : ctx(ctx),
      table_(std::move(tab)),
      base_arrow_index_(std::make_shared<cylon::ArrowRangeIndex>(0, table_->num_rows(), 1,
                                                                 cylon::ToArrowPool(ctx))) {}

#ifdef BUILD_CYLON_PARQUET
Status FromParquet(const std::shared_ptr<CylonContext> &ctx, const std::string &path,
                   std::shared_ptr<Table> &tableOut) {
  arrow::Result<std::shared_ptr<arrow::Table>> result = cylon::io::ReadParquet(ctx, path);
  if (result.ok()) {
    std::shared_ptr<arrow::Table> table = result.ValueOrDie();
    LOG(INFO) << "Chunks " << table->column(0)->chunks().size();
    if (table->column(0)->chunks().size() > 1) {
      auto combine_res = table->CombineChunks(ToArrowPool(ctx));
      if (!combine_res.ok()) {
        return Status(static_cast<int>(combine_res.status().code()),
                      combine_res.status().message());
      }
      tableOut = std::make_shared<Table>(ctx, combine_res.ValueOrDie());
    } else {
      tableOut = std::make_shared<Table>(ctx, table);
    }
    return Status::OK();
  }
  return Status(Code::IOError, result.status().message());
}

void ReadParquetThread(const std::shared_ptr<CylonContext> &ctx, const std::string &path,
                       std::shared_ptr<cylon::Table> *table,
                       const std::shared_ptr<std::promise<Status>> &status_promise) {
  status_promise->set_value(FromParquet(ctx, path, *table));
}

Status FromParquet(const std::shared_ptr<CylonContext> &ctx, const std::vector<std::string> &paths,
                   const std::vector<std::shared_ptr<Table> *> &tableOuts,
                   const io::config::ParquetOptions &options) {
  if (options.IsConcurrentFileReads()) {
    std::vector<std::pair<std::future<Status>, std::thread>> futures;
    futures.reserve(paths.size());
    for (uint64_t kI = 0; kI < paths.size(); ++kI) {
      auto read_promise = std::make_shared<std::promise<Status>>();
      futures.emplace_back(
          read_promise->get_future(),
          std::thread(ReadParquetThread, std::cref(ctx), std::cref(paths[kI]), tableOuts[kI],
                      read_promise));
    }
    bool all_passed = true;
    for (auto &future: futures) {
      auto status = future.first.get();
      all_passed &= status.is_ok();
      future.second.join();
    }
    return all_passed ? Status::OK() : Status(cylon::IOError, "Failed to read the parquet files");
  } else {
    auto status = Status::OK();
    for (std::size_t kI = 0; kI < paths.size(); ++kI) {
      status = FromParquet(ctx, paths[kI], *tableOuts[kI]);
      if (!status.is_ok()) {
        return status;
      }
    }
    return status;
  }
}

Status WriteParquet(const std::shared_ptr<cylon::CylonContext> &ctx_,
                    std::shared_ptr<cylon::Table> &table,
                    const std::string &path,
                    const io::config::ParquetOptions &options) {
  arrow::Status writefile_result = cylon::io::WriteParquet(ctx_, table, path, options);
  if (!writefile_result.ok()) {
    return Status(Code::IOError, writefile_result.message());
  }

  return Status(Code::OK);
}
#endif
}  // namespace cylon
