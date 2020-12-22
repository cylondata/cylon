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
#include "table.hpp"

#include <arrow/table.h>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <arrow/compute/api.h>
#include <future>
#include <join/join_utils.hpp>

#include "table_api_extended.hpp"
#include "io/arrow_io.hpp"
#include "join/join.hpp"
#include  "util/to_string.hpp"
#include "iostream"
#include "util/arrow_utils.hpp"
#include "util/uuid.hpp"
#include "arrow/arrow_all_to_all.hpp"
#include "arrow/arrow_comparator.hpp"
#include "ctx/arrow_memory_pool_utils.hpp"
#include "arrow/arrow_types.hpp"
#include "partition/partition.hpp"

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
                    const std::shared_ptr<arrow::Table> &table,
                    const int32_t col_idx,
                    const std::shared_ptr<std::vector<int64_t>> &row_indices,
                    arrow::ArrayVector &array_vector) {
  std::shared_ptr<arrow::Array> destination_col_array;
  arrow::Status ar_status = cylon::util::copy_array_by_indices(*row_indices,
                                                               table->column(col_idx)->chunk(0),
                                                               &destination_col_array, cylon::ToArrowPool(ctx));
  if (ar_status != arrow::Status::OK()) {
    LOG(FATAL) << "Failed while copying a column to the final table from tables."
               << ar_status.ToString();
    return Status(static_cast<int>(ar_status.code()), ar_status.message());
  }
  array_vector.push_back(destination_col_array);
  return Status::OK();
}

static inline Status all_to_all_arrow_tables(std::shared_ptr<CylonContext> &ctx,
                                             const std::shared_ptr<arrow::Schema> &schema,
                                             const std::vector<std::shared_ptr<arrow::Table>> &partitioned_tables,
                                             std::shared_ptr<arrow::Table> &table_out) {
  const auto &neighbours = ctx->GetNeighbours(true);
  std::vector<std::shared_ptr<arrow::Table>> received_tables;
  received_tables.reserve(neighbours.size());

  // define call back to catch the receiving tables
  ArrowCallback arrow_callback =
      [&received_tables](int source, const std::shared_ptr<arrow::Table> &table_, int reference) {
        received_tables.push_back(table_);
        return true;
      };

  // doing all to all communication to exchange tables
  cylon::ArrowAllToAll all_to_all(ctx, neighbours, neighbours, ctx->GetNextSequence(), arrow_callback, schema);

  // if world size == partitions, simply send paritions based on index
  const size_t world_size = (size_t) ctx->GetWorldSize(), num_partitions = partitioned_tables.size(),
      rank = ctx->GetRank();
  if (world_size == num_partitions) {
    for (size_t i = 0; i < partitioned_tables.size(); i++) {
      if (i != rank) {
        all_to_all.insert(partitioned_tables[i], i);
      } else {
        received_tables.push_back(partitioned_tables[i]);
      }
    }
  } else { // divide parititions to world_size potions and send accordingly
    for (size_t i = 0; i < partitioned_tables.size(); i++) {
      size_t target = i * world_size / num_partitions;
      if (target != rank) {
        all_to_all.insert(partitioned_tables[i], target);
      } else {
        received_tables.push_back(partitioned_tables[i]);
      }
    }
  }

  // now complete the communication
  all_to_all.finish();
  while (!all_to_all.isComplete()) {}
  all_to_all.close();

/*  // now clear locally partitioned tables
  partitioned_tables.clear();*/

  // now we have the final set of tables
  LOG(INFO) << "Concatenating tables, Num of tables :  " << received_tables.size();
  arrow::Result<std::shared_ptr<arrow::Table>> concat_res = arrow::ConcatenateTables(received_tables);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(concat_res.status())
  const auto &final_table = concat_res.ValueOrDie();
  LOG(INFO) << "Done concatenating tables, rows :  " << final_table->num_rows();

  arrow::Result<std::shared_ptr<arrow::Table>> combine_res = final_table->CombineChunks(cylon::ToArrowPool(ctx));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(concat_res.status())
  table_out = combine_res.ValueOrDie();

  return Status::OK();
}

template<typename T>
// T is int32_t or const std::vector<int32_t>&
static inline Status shuffle_table_by_hashing(std::shared_ptr<CylonContext> &ctx,
                                              std::shared_ptr<Table> &table,
                                              T hash_column,
                                              std::shared_ptr<arrow::Table> &table_out) {
  // partition the tables locally
  std::vector<uint32_t> outPartitions, counts;
  int no_of_partitions = ctx->GetWorldSize();
  auto status = PartitionByHashing(table, hash_column, no_of_partitions, outPartitions, counts);
  LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(status)

  std::vector<std::shared_ptr<arrow::Table>> partitioned_tables;
  status = Split(table, no_of_partitions, outPartitions, counts, partitioned_tables);
  LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(status)

  std::shared_ptr<arrow::Schema> schema = table->get_table()->schema();
  // we are going to free if retain is set to false
  if (!table->IsRetain()) {
    table.reset();
  }

  return all_to_all_arrow_tables(ctx, schema, partitioned_tables, table_out);
}

template<typename T>
// T is int32_t or const std::vector<int32_t>&
static inline Status shuffle_two_tables_by_hashing(std::shared_ptr<cylon::CylonContext> &ctx,
                                                   std::shared_ptr<cylon::Table> &left_table,
                                                   T left_hash_column,
                                                   std::shared_ptr<cylon::Table> &right_table,
                                                   T right_hash_column,
                                                   std::shared_ptr<arrow::Table> &left_table_out,
                                                   std::shared_ptr<arrow::Table> &right_table_out) {
  LOG(INFO) << "Shuffling two tables with total rows : " << left_table->Rows() + right_table->Rows();
  auto t1 = std::chrono::high_resolution_clock::now();
  auto status = shuffle_table_by_hashing(ctx, left_table, left_hash_column, left_table_out);
  RETURN_CYLON_STATUS_IF_FAILED(status)

  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Left shuffle time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  status = shuffle_table_by_hashing(ctx, right_table, right_hash_column, right_table_out);
  RETURN_CYLON_STATUS_IF_FAILED(status)

  auto t3 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Right shuffle time : " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

  return status;
}

Status FromCSV(std::shared_ptr<cylon::CylonContext> &ctx,
               const std::string &path,
               std::shared_ptr<Table> &tableOut,
               const cylon::io::config::CSVReadOptions &options) {
  arrow::Result<std::shared_ptr<arrow::Table>> result = cylon::io::read_csv(ctx, path, options);
  if (result.ok()) {
    std::shared_ptr<arrow::Table> table = result.ValueOrDie();
    LOG(INFO) << "Chunks " << table->column(0)->chunks().size();
    if (table->column(0)->chunks().size() > 1) {
      auto combine_res = table->CombineChunks(ToArrowPool(ctx));
      if (!combine_res.ok()) {
        return Status(static_cast<int>(combine_res.status().code()), combine_res.status().message());
      }
      tableOut = std::make_shared<Table>(combine_res.ValueOrDie(), ctx);
    } else {
      tableOut = std::make_shared<Table>(table, ctx);
    }
    return Status::OK();
  }
  return Status(Code::IOError, result.status().message());
}

Status Table::FromArrowTable(std::shared_ptr<cylon::CylonContext> &ctx,
                             std::shared_ptr<arrow::Table> &table,
                             std::shared_ptr<Table> &tableOut) {
  if (!cylon::tarrow::validateArrowTableTypes(table)) {
    LOG(FATAL) << "Types not supported";
    return Status(cylon::Invalid, "This type not supported");
  }
  tableOut = std::make_shared<Table>(table, ctx);
  return Status(cylon::OK, "Loaded Successfully");
}

Status Table::FromColumns(std::shared_ptr<cylon::CylonContext> &ctx,
                          std::vector<std::shared_ptr<Column>> &&columns,
                          std::shared_ptr<Table> &tableOut) {
  arrow::Status status;
  arrow::SchemaBuilder schema_builder;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> col_arrays;
  col_arrays.reserve(columns.size());

  std::shared_ptr<cylon::DataType> data_type;
  for (const std::shared_ptr<Column> &col: columns) {
    data_type = col->GetDataType();
    auto field = arrow::field(col->GetID(), cylon::tarrow::convertToArrowType(data_type));
    status = schema_builder.AddField(field);

    if (!status.ok()) return Status(Code::UnknownError, status.message());
    col_arrays.push_back(col->GetColumnData());
  }

  auto schema_result = schema_builder.Finish();
  std::shared_ptr<arrow::Table> arrow_table = arrow::Table::Make(schema_result.ValueOrDie(), col_arrays);

  if (!cylon::tarrow::validateArrowTableTypes(arrow_table)) {
    LOG(FATAL) << "Types not supported";
    return Status(cylon::Invalid, "This type not supported");
  }
  tableOut = std::make_shared<Table>(arrow_table, ctx, columns);

  return Status(cylon::OK, "Loaded Successfully");
}

Status WriteCSV(std::shared_ptr<cylon::Table> &table,
                const std::string &path,
                const cylon::io::config::CSVWriteOptions &options) {
  std::ofstream out_csv;
  out_csv.open(path);
  Status status = table->PrintToOStream(0, table->get_table()->num_columns(),
                                        0, table->get_table()->num_rows(),
                                        out_csv,
                                        options.GetDelimiter(),
                                        options.IsOverrideColumnNames(),
                                        options.GetColumnNames());
  out_csv.close();
  return status;
}

int Table::Columns() {
  return table_->num_columns();
}

std::vector<std::string> Table::ColumnNames() {
  return table_->ColumnNames();
}

int64_t Table::Rows() {
  return table_->num_rows();
}

void Table::Print() {
  Print(0, this->Rows(), 0, this->Columns());
}

void Table::Print(int row1, int row2, int col1, int col2) {
  PrintToOStream(col1, col2, row1, row2, std::cout);
}

Status Merge(std::shared_ptr<cylon::CylonContext> &ctx,
             const std::vector<std::shared_ptr<cylon::Table>> &ctables,
             std::shared_ptr<Table> &tableOut) {
  std::vector<std::shared_ptr<arrow::Table>> tables;
  for (auto it = ctables.begin(); it < ctables.end(); it++) {
    std::shared_ptr<arrow::Table> arrow;
    (*it)->ToArrowTable(arrow);
    tables.push_back(arrow);
  }
  arrow::Result<std::shared_ptr<arrow::Table>> concat_res = arrow::ConcatenateTables(tables);
  if (concat_res.ok()) {
    arrow::Result<std::shared_ptr<arrow::Table>>
        combined_res = concat_res.ValueOrDie()->CombineChunks(cylon::ToArrowPool(ctx));
    if (!combined_res.ok()) {
      return Status(static_cast<int>(combined_res.status().code()), combined_res.status().message());
    }
    tableOut = std::make_shared<cylon::Table>(combined_res.ValueOrDie(), ctx);
    return Status::OK();
  } else {
    return Status(static_cast<int>(concat_res.status().code()), concat_res.status().message());
  }
}

Status Sort(std::shared_ptr<cylon::Table> &table, int sort_column, std::shared_ptr<cylon::Table> &out) {
  std::shared_ptr<arrow::Table> sorted_table;
  auto table_ = table->get_table();
  auto ctx = table->GetContext();
  arrow::Status status = cylon::util::SortTable(table_, sort_column, cylon::ToArrowPool(ctx), sorted_table);
  if (status.ok()) {
    return Table::FromArrowTable(ctx, sorted_table, out);
  } else {
    return Status(static_cast<int>(status.code()), status.message());
  }
}

Status DistributedSort(std::shared_ptr<cylon::Table> &table,
                       int sort_column,
                       std::shared_ptr<Table> &output,
                       SortOptions sort_options) {
  auto ctx = table->GetContext();
  int world_sz = ctx->GetWorldSize();

  std::shared_ptr<arrow::Table> arrow_table, sorted_table;
  // first do distributed sort partitioning
  if (world_sz == 1) {
    arrow_table = table->get_table();
  } else {
    std::vector<uint32_t> target_partitions, partition_hist;
    std::vector<std::shared_ptr<arrow::Table>> split_tables;

    Status status = PartitionBySorting(table,
                                       sort_column,
                                       world_sz,
                                       target_partitions,
                                       partition_hist,
                                       sort_options.ascending,
                                       sort_options.num_samples,
                                       sort_options.num_bins);
    RETURN_CYLON_STATUS_IF_FAILED(status)

    status = Split(table, world_sz, target_partitions, partition_hist, split_tables);
    RETURN_CYLON_STATUS_IF_FAILED(status)

    // we are going to free if retain is set to false
    std::shared_ptr<arrow::Schema> schema = table->get_table()->schema();
    if (!table->IsRetain()) {
      table.reset();
    }

    status = all_to_all_arrow_tables(ctx, schema, split_tables, arrow_table);
    RETURN_CYLON_STATUS_IF_FAILED(status)
  }

  //then do a local sort
  auto astatus = util::SortTable(arrow_table, sort_column, ToArrowPool(ctx), sorted_table);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(astatus)

  return Table::FromArrowTable(ctx, sorted_table, output);
}

Status HashPartition(std::shared_ptr<cylon::Table> &table, const std::vector<int> &hash_columns, int no_of_partitions,
                     std::unordered_map<int, std::shared_ptr<cylon::Table>> *out) {
  // keep arrays for each target, these arrays are used for creating the table
  std::vector<uint32_t> outPartitions, counts;
  auto status = PartitionByHashing(table, hash_columns, no_of_partitions, outPartitions, counts);
  LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(status)

  std::vector<std::shared_ptr<arrow::Table>> partitioned_tables;
  status = Split(table, no_of_partitions, outPartitions, counts, partitioned_tables);
  LOG_AND_RETURN_CYLON_STATUS_IF_FAILED(status)

  auto ctx = table->GetContext();
  for (int i = 0; i < no_of_partitions; i++) {
    out->insert(std::make_pair(i, std::make_shared<Table>(partitioned_tables[i], ctx)));
  }

  return Status::OK();
}

arrow::Status create_table_with_duplicate_index(arrow::MemoryPool *pool,
                                                std::shared_ptr<arrow::Table> &table,
                                                size_t index_column,
                                                std::vector<std::shared_ptr<arrow::ChunkedArray>> &nl_vectors) {
  const std::vector<std::shared_ptr<arrow::ChunkedArray>> &chunk_arrays = table->columns();
  for (size_t i = 0; i < chunk_arrays.size(); i++) {
    if (i != index_column) {
      nl_vectors.push_back(chunk_arrays[i]);
    } else {
      std::shared_ptr<arrow::ChunkedArray> new_c_array;
      arrow::Status st = cylon::util::duplicate(chunk_arrays[i],
                                                table->schema()->fields()[i],
                                                pool,
                                                new_c_array);
      if (st != arrow::Status::OK()) {
        return st;
      }
      nl_vectors.push_back(new_c_array);
    }
  }
  table = arrow::Table::Make(table->schema(), nl_vectors);
  return arrow::Status::OK();
}

Status Join(std::shared_ptr<cylon::Table> &left, std::shared_ptr<cylon::Table> &right,
            cylon::join::config::JoinConfig join_config,
            std::shared_ptr<cylon::Table> &out) {
  if (left == NULLPTR) {
    return Status(Code::KeyError, "Couldn't find the left table");
  } else if (right == NULLPTR) {
    return Status(Code::KeyError, "Couldn't find the right table");
  } else {
    std::shared_ptr<arrow::Table> table;
    std::shared_ptr<arrow::Table> left_table;
    std::shared_ptr<arrow::Table> right_table;
    auto ctx = left->GetContext();
    left->ToArrowTable(left_table);
    right->ToArrowTable(right_table);
    // if it is a sort algorithm and certian key types, we are going to do an in-place sort
    if (join_config.GetAlgorithm() == cylon::join::config::SORT) {
      size_t lIndex = join_config.GetLeftColumnIdx();
      size_t rIndex = join_config.GetRightColumnIdx();
      auto left_type = left_table->column(lIndex)->type()->id();
      if (cylon::join::util::is_inplace_join_possible(left_type)) {
        // now create a copy
        std::vector<std::shared_ptr<arrow::ChunkedArray>> nl_vectors;
        std::vector<std::shared_ptr<arrow::ChunkedArray>> nr_vectors;
        // we don't have to copy if the table is freed
        if (!left->IsRetain()) {
          arrow::Status st = create_table_with_duplicate_index(cylon::ToArrowPool(ctx),
                                                               left_table, lIndex, nl_vectors);
          if (st != arrow::Status::OK()) {
            return Status(static_cast<int>(st.code()), st.message());
          }
        }
        if (!right->IsRetain()) {
          arrow::Status st = create_table_with_duplicate_index(cylon::ToArrowPool(ctx),
                                                               right_table, rIndex, nr_vectors);
          if (st != arrow::Status::OK()) {
            return Status(static_cast<int>(st.code()), st.message());
          }
        }
      }
    }

    arrow::Status status = join::joinTables(left_table,
                                            right_table,
                                            join_config,
                                            &table,
                                            cylon::ToArrowPool(ctx));
    if (status == arrow::Status::OK()) {
      out = std::make_shared<cylon::Table>(table, ctx);
    }
    return Status(static_cast<int>(status.code()), status.message());
  }
}

Status Table::ToArrowTable(std::shared_ptr<arrow::Table> &out) {
  out = table_;
  return Status::OK();
}

Status DistributedJoin(std::shared_ptr<cylon::Table> &left,
                       std::shared_ptr<cylon::Table> &right,
                       cylon::join::config::JoinConfig join_config,
                       std::shared_ptr<cylon::Table> &out) {
  // check whether the world size is 1
  std::shared_ptr<cylon::CylonContext> ctx = left->GetContext();
  if (ctx->GetWorldSize() == 1) {
    return Join(left, right, join_config, out);
  }

  std::shared_ptr<arrow::Table> left_final_table, right_final_table;
  auto shuffle_status = shuffle_two_tables_by_hashing(ctx,
                                                      left,
                                                      join_config.GetLeftColumnIdx(),
                                                      right,
                                                      join_config.GetRightColumnIdx(),
                                                      left_final_table,
                                                      right_final_table);
  RETURN_CYLON_STATUS_IF_FAILED(shuffle_status)

  std::shared_ptr<arrow::Table> table;
  arrow::Status status = join::joinTables(left_final_table,
                                          right_final_table,
                                          join_config,
                                          &table,
                                          cylon::ToArrowPool(ctx));
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(status)
  out = std::make_shared<cylon::Table>(table, ctx);

  return Status::OK();
}

Status Select(std::shared_ptr<cylon::Table> &table,
              const std::function<bool(cylon::Row)> &selector,
              std::shared_ptr<Table> &out) {
  // boolean builder to hold the mask
  auto ctx = table->GetContext();
  auto table_ = table->get_table();
  auto row = cylon::Row(table_);
  arrow::BooleanBuilder boolean_builder(cylon::ToArrowPool(ctx));
  auto kI = table->Rows();
  arrow::Status status = boolean_builder.Reserve(kI);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(status)

  for (int64_t row_index = 0; row_index < kI; row_index++) {
    row.SetIndex(row_index);
    boolean_builder.UnsafeAppend(selector(row));
  }
  // building the mask
  std::shared_ptr<arrow::Array> mask;
  status = boolean_builder.Finish(&mask);
  RETURN_CYLON_STATUS_IF_ARROW_FAILED(status)

  arrow::Result<arrow::Datum> filter_res = arrow::compute::Filter(arrow::Datum(table_), arrow::Datum(mask));
  if (!filter_res.ok()) {
    status = filter_res.status();
    return Status(static_cast<int>(status.code()), status.message());
  }
  std::shared_ptr<arrow::Table> out_table = filter_res.ValueOrDie().table();
  out = std::make_shared<cylon::Table>(out_table, ctx);
  return Status::OK();
}

Status Union(std::shared_ptr<Table> &first, std::shared_ptr<Table> &second,
             std::shared_ptr<Table> &out) {
  std::shared_ptr<arrow::Table> ltab = first->get_table();
  std::shared_ptr<arrow::Table> rtab = second->get_table();
  Status status = VerifyTableSchema(ltab, second->get_table());
  if (!status.is_ok()) return status;
  int64_t eq_calls = 0, hash_calls = 0;
  auto ctx = first->GetContext();
  std::shared_ptr<arrow::Table> tables[2] = {ltab, second->get_table()};
  RowComparator row_comp(ctx, tables, &eq_calls, &hash_calls);
//  std::vector<std::shared_ptr<arrow::Table>> tables{ltab, second->get_table()};
//  MultiTableRowIndexComparator row_comp(tables);
//  MultiTableRowIndexHash row_hash(tables);

  auto buckets_pre_alloc = (ltab->num_rows() + rtab->num_rows());
  LOG(INFO) << "Buckets : " << buckets_pre_alloc;
  std::unordered_set<std::pair<int8_t, int64_t>, RowComparator, RowComparator>
      rows_set(buckets_pre_alloc, row_comp, row_comp);
  const int64_t max = std::max(ltab->num_rows(), rtab->num_rows());
  const int8_t table0 = 0;
  const int8_t table1 = 1;
  const int64_t print_threshold = max / 10;
  for (int64_t row = 0; row < max; ++row) {
    if (row < ltab->num_rows()) {
      rows_set.insert(std::pair<int8_t, int64_t>(table0, row));
    }

    if (row < rtab->num_rows()) {
      rows_set.insert(std::pair<int8_t, int64_t>(table1, row));
    }

    if (row % print_threshold == 0) {
      LOG(INFO) << "Done " << (row + 1) * 100 / max << "%" << " N : "
                << row << ", Eq : " << eq_calls << ", Hs : "
                << hash_calls;
    }
  }

  std::shared_ptr<std::vector<int64_t>> indices_from_tabs[2] = {
      std::make_shared<std::vector<int64_t>>(),
      std::make_shared<std::vector<int64_t>>()
  };

  for (auto const &pr : rows_set) {
    indices_from_tabs[pr.first]->push_back(pr.second);
  }
  std::vector<std::shared_ptr<arrow::ChunkedArray>> final_data_arrays;
  // prepare final arrays
  for (int32_t col_idx = 0; col_idx < ltab->num_columns(); col_idx++) {
    arrow::ArrayVector array_vector;
    for (int tab_idx = 0; tab_idx < 2; tab_idx++) {
      status = PrepareArray(ctx, tables[tab_idx], col_idx, indices_from_tabs[tab_idx], array_vector);

      if (!status.is_ok()) return status;
    }
    final_data_arrays.push_back(std::make_shared<arrow::ChunkedArray>(array_vector));
  }
  // create final table
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(ltab->schema(), final_data_arrays);
  arrow::Result<std::shared_ptr<arrow::Table>> merge_res = table->CombineChunks(cylon::ToArrowPool(ctx));
  if (!merge_res.ok()) {
    return Status(static_cast<int>(merge_res.status().code()), merge_res.status().message());
  }
  out = std::make_shared<cylon::Table>(merge_res.ValueOrDie(), ctx);
  return Status::OK();
}

Status Subtract(std::shared_ptr<Table> &first,
                std::shared_ptr<Table> &second, std::shared_ptr<Table> &out) {
  std::shared_ptr<arrow::Table> ltab = first->get_table();
  std::shared_ptr<arrow::Table> rtab = second->get_table();
  Status status = VerifyTableSchema(ltab, rtab);
  auto ctx = first->GetContext();
  if (!status.is_ok()) {
    return status;
  }
  std::shared_ptr<arrow::Table> tables[2] = {ltab, rtab};
  int64_t eq_calls = 0, hash_calls = 0;
  auto row_comp = RowComparator(ctx, tables, &eq_calls, &hash_calls);
  auto buckets_pre_alloc = ltab->num_rows();
  LOG(INFO) << "Buckets : " << buckets_pre_alloc;
  std::unordered_set<std::pair<int8_t, int64_t>, RowComparator, RowComparator>
      left_row_set(buckets_pre_alloc, row_comp, row_comp);
  auto t1 = std::chrono::steady_clock::now();
  // first populate left table in the hash set
  int64_t print_offset = ltab->num_rows() / 4, next_print = print_offset;
  for (int64_t row = 0; row < ltab->num_rows(); ++row) {
    left_row_set.insert(std::pair<int8_t, int64_t>(0, row));

    if (row == next_print) {
      LOG(INFO) << "Done " << (row + 1) * 100 / ltab->num_rows() << "%"
                << " N : " << row << ", Eq : " << eq_calls
                << ", Hs : " << hash_calls;
      next_print += print_offset;
    }
  }
  // then remove matching rows using hash map comparator
  print_offset = rtab->num_rows() / 4;
  next_print = print_offset;
  for (int64_t row = 0; row < rtab->num_rows(); ++row) {
    // finds a matching row from left and erase it
    left_row_set.erase(std::pair<int8_t, int64_t>(1, row));
    if (row == next_print) {
      LOG(INFO) << "Done " << (row + 1) * 100 / rtab->num_rows() << "%" << " N : "
                << row << ", Eq : " << eq_calls
                << ", Hs : " << hash_calls;
      next_print += print_offset;
    }
  }

  auto t2 = std::chrono::steady_clock::now();
  LOG(INFO) << "Adding to Set took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms";
  std::shared_ptr<std::vector<int64_t>> left_indices = std::make_shared<std::vector<int64_t>>();
  left_indices->reserve(left_row_set.size());  // reserve space for vec
  for (auto const &pr : left_row_set) {
    left_indices->push_back(pr.second);
  }
  std::vector<std::shared_ptr<arrow::ChunkedArray>> final_data_arrays;
  t1 = std::chrono::steady_clock::now();
  // prepare final arrays
  for (int32_t col_idx = 0; col_idx < ltab->num_columns(); col_idx++) {
    arrow::ArrayVector array_vector;
    status = PrepareArray(ctx, ltab, col_idx, left_indices, array_vector);
    if (!status.is_ok()) {
      return status;
    }
    final_data_arrays.push_back(std::make_shared<arrow::ChunkedArray>(array_vector));
  }
  t2 = std::chrono::steady_clock::now();
  LOG(INFO) << "Final array preparation took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
            << "ms";
  // create final table
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(ltab->schema(), final_data_arrays);
  out = std::make_shared<cylon::Table>(table, ctx);
  return Status::OK();
}

Status Intersect(std::shared_ptr<Table> &first,
                 std::shared_ptr<Table> &second, std::shared_ptr<Table> &out) {
  std::shared_ptr<arrow::Table> ltab = first->get_table();
  std::shared_ptr<arrow::Table> rtab = second->get_table();
  Status status = VerifyTableSchema(ltab, rtab);
  auto ctx = first->GetContext();
  if (!status.is_ok()) {
    return status;
  }
  std::shared_ptr<arrow::Table> tables[2] = {ltab, rtab};
  int64_t eq_calls = 0, hash_calls = 0;
  auto row_comp = RowComparator(ctx, tables, &eq_calls, &hash_calls);
  auto buckets_pre_alloc = (ltab->num_rows() + rtab->num_rows());
  LOG(INFO) << "Buckets : " << buckets_pre_alloc;
  std::unordered_set<std::pair<int8_t, int64_t>, RowComparator, RowComparator>
      rows_set(buckets_pre_alloc, row_comp, row_comp);
  auto t1 = std::chrono::steady_clock::now();
  // first populate left table in the hash set
  int64_t print_offset = ltab->num_rows() / 4, next_print = print_offset;
  for (int64_t row = 0; row < ltab->num_rows(); ++row) {
    rows_set.insert(std::pair<int8_t, int64_t>(0, row));
    if (row == next_print) {
      LOG(INFO) << "left done " << (row + 1) * 100 / ltab->num_rows() << "%" << " N : "
                << row << ", Eq : " << eq_calls
                << ", Hs : " << hash_calls;
      next_print += print_offset;
    }
  }

  std::unordered_set<int64_t> left_indices_set(ltab->num_rows());
  // then add matching rows to the indices_from_tabs vector
  print_offset = rtab->num_rows() / 4;
  next_print = print_offset;
  for (int64_t row = 0; row < rtab->num_rows(); ++row) {
    auto matching_row_it = rows_set.find(std::pair<int8_t, int64_t>(1, row));
    if (matching_row_it != rows_set.end()) {
      std::pair<int8_t, int64_t> matching_row = *matching_row_it;
      left_indices_set.insert(matching_row.second);
    }
    if (row == next_print) {
      LOG(INFO) << "right done " << (row + 1) * 100 / rtab->num_rows()
                << "%" << " N : " << row << ", Eq : " << eq_calls
                << ", Hs : " << hash_calls;
      next_print += print_offset;
    }
  }
  // convert set to vector todo: find a better way to do this inplace!
  std::shared_ptr<std::vector<int64_t>> left_indices = std::make_shared<std::vector<int64_t>>();
  left_indices->reserve(left_indices_set.size() / 2);
  left_indices->assign(left_indices_set.begin(), left_indices_set.end());
  left_indices->shrink_to_fit();
  auto t2 = std::chrono::steady_clock::now();
  LOG(INFO) << "Adding to Set took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms";
  std::vector<std::shared_ptr<arrow::ChunkedArray>> final_data_arrays;
  t1 = std::chrono::steady_clock::now();
  // prepare final arrays
  for (int32_t col_idx = 0; col_idx < ltab->num_columns(); col_idx++) {
    arrow::ArrayVector array_vector;
    status = PrepareArray(ctx, ltab, col_idx, left_indices, array_vector);

    if (!status.is_ok()) return status;

    final_data_arrays.push_back(std::make_shared<arrow::ChunkedArray>(array_vector));
  }

  t2 = std::chrono::steady_clock::now();
  LOG(INFO) << "Final array preparation took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
            << "ms";
  // create final table
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(ltab->schema(), final_data_arrays);
  out = std::make_shared<cylon::Table>(table, ctx);
  return Status::OK();
}

typedef Status
(*LocalSetOperation)(std::shared_ptr<cylon::Table> &, std::shared_ptr<cylon::Table> &,
                     std::shared_ptr<cylon::Table> &);

Status DoDistributedSetOperation(LocalSetOperation local_operation,
                                 std::shared_ptr<cylon::Table> &table_left,
                                 std::shared_ptr<cylon::Table> &table_right,
                                 std::shared_ptr<cylon::Table> &out) {
  // extract the tables out
  auto left = table_left->get_table();
  auto right = table_right->get_table();
  auto ctx = table_left->GetContext();

  Status status = VerifyTableSchema(left, right);
  if (!status.is_ok()) {
    return status;
  }

  if (ctx->GetWorldSize() < 2) {
    return local_operation(table_left, table_right, out);
  }

  std::vector<int32_t> hash_columns;
  hash_columns.reserve(left->num_columns());
  for (int kI = 0; kI < left->num_columns(); ++kI) {
    hash_columns.push_back(kI);
  }

  std::shared_ptr<arrow::Table> left_final_table;
  std::shared_ptr<arrow::Table> right_final_table;
  auto shuffle_status = shuffle_two_tables_by_hashing(ctx,
                                                      table_left,
                                                      hash_columns,
                                                      table_right,
                                                      hash_columns,
                                                      left_final_table,
                                                      right_final_table);
  if (shuffle_status.is_ok()) {
    std::shared_ptr<cylon::Table> left_tab = std::make_shared<cylon::Table>(left_final_table, ctx);
    std::shared_ptr<cylon::Table> right_tab = std::make_shared<cylon::Table>(right_final_table, ctx);
    // now do the local union
    std::shared_ptr<arrow::Table> table;
    status = local_operation(left_tab, right_tab, out);
    return status;
  } else {
    return shuffle_status;
  }
}

Status DistributedUnion(std::shared_ptr<Table> &left, std::shared_ptr<Table> &right,
                        std::shared_ptr<Table> &out) {
  auto ctx = left->GetContext();
  return DoDistributedSetOperation(&Union, left, right, out);
}

Status DistributedSubtract(std::shared_ptr<Table> &left, std::shared_ptr<Table> &right,
                           std::shared_ptr<Table> &out) {
  auto ctx = left->GetContext();
  return DoDistributedSetOperation(&Subtract, left, right, out);
}

Status DistributedIntersect(std::shared_ptr<Table> &left, std::shared_ptr<Table> &right,
                            std::shared_ptr<Table> &out) {
  auto ctx = left->GetContext();
  return DoDistributedSetOperation(&Intersect, left, right, out);
}

void Table::Clear() {
}

Table::~Table() {
  this->Clear();
}

void ReadCSVThread(const std::shared_ptr<CylonContext> &ctx, const std::string &path,
                   std::shared_ptr<cylon::Table> *table,
                   const cylon::io::config::CSVReadOptions &options,
                   const std::shared_ptr<std::promise<Status>> &status_promise) {
  std::shared_ptr<CylonContext> ctx_ = ctx; // make a copy of the shared ptr
  status_promise->set_value(FromCSV(ctx_,
                                    path,
                                    *table,
                                    options));
}

Status FromCSV(std::shared_ptr<cylon::CylonContext> &ctx, const std::vector<std::string> &paths,
               const std::vector<std::shared_ptr<Table> *> &tableOuts,
               io::config::CSVReadOptions options) {
  if (options.IsConcurrentFileReads()) {
    std::vector<std::pair<std::future<Status>, std::thread>> futures;
    futures.reserve(paths.size());
    for (uint64_t kI = 0; kI < paths.size(); ++kI) {
      auto read_promise = std::make_shared<std::promise<Status>>();
//	  auto context = ctx.get();
      futures.emplace_back(read_promise->get_future(),
                           std::thread(ReadCSVThread,
                                       ctx,
                                       paths[kI],
                                       tableOuts[kI],
                                       options,
                                       read_promise));
    }
    bool all_passed = true;
    for (auto &future : futures) {
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

Status Project(std::shared_ptr<cylon::Table> &table,
               const std::vector<int64_t> &project_columns,
               std::shared_ptr<Table> &out) {
  std::vector<std::shared_ptr<arrow::Field>> schema_vector;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> column_arrays;
  schema_vector.reserve(project_columns.size());
  auto table_ = table->get_table();
  auto ctx = table->GetContext();

  for (auto const &col_index : project_columns) {
    schema_vector.push_back(table_->field(col_index));
    auto chunked_array = std::make_shared<arrow::ChunkedArray>(table_->column(col_index)->chunks());
    column_arrays.push_back(chunked_array);
  }

  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  std::shared_ptr<arrow::Table> ar_table = arrow::Table::Make(schema, column_arrays);
  out = std::make_shared<cylon::Table>(ar_table, ctx);
  return Status::OK();
}

std::shared_ptr<cylon::CylonContext> Table::GetContext() {
  return this->ctx;
}

Status Table::PrintToOStream(
    int col1,
    int col2,
    int row1,
    int row2,
    std::ostream &out,
    char delimiter,
    bool use_custom_header,
    const std::vector<std::string> &headers) {
  auto table = table_;
  if (table != NULLPTR) {
    // print the headers
    if (use_custom_header) {
      // check if the headers are valid
      if (headers.size() != (uint64_t) table->num_columns()) {
        return Status(cylon::Code::IndexError,
                      "Provided headers doesn't match with the number of columns of the table. Given "
                          + std::to_string(headers.size()) + ", Expected "
                          + std::to_string(table->num_columns()));
      }

      for (int col = col1; col < col2; col++) {
        out << headers[col];
        if (col != col2 - 1) {
          out << delimiter;
        } else {
          out << std::endl;
        }
      }
    }
    for (int row = row1; row < row2; row++) {
      for (int col = col1; col < col2; col++) {
        auto column = table->column(col);
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

std::shared_ptr<arrow::Table> Table::get_table() {
  return table_;
}

bool Table::IsRetain() const {
  return retain_;
}

std::shared_ptr<Column> Table::GetColumn(int32_t index) const {
  return this->columns_.at(index);
}

std::vector<std::shared_ptr<cylon::Column>> Table::GetColumns() const {
  return this->columns_;
}

Status Shuffle(std::shared_ptr<cylon::Table> &table,
               const std::vector<int> &hash_columns,
               std::shared_ptr<cylon::Table> &output) {
  auto ctx_ = table->GetContext();
  std::shared_ptr<arrow::Table> table_out;
  cylon::Status status = shuffle_table_by_hashing(ctx_, table, hash_columns, table_out);

  if (!status.is_ok()) {
    LOG(FATAL) << "table shuffle failed!";
    return status;
  }

  return cylon::Table::FromArrowTable(ctx_, table_out, output);
}

Status Unique(std::shared_ptr<cylon::Table> &in,
              const std::vector<int> &cols,
              std::shared_ptr<cylon::Table> &out,
              bool first) {
  auto p1 = std::chrono::high_resolution_clock::now();
  std::shared_ptr<arrow::Table> ltab = in->get_table();
  //int64_t eq_calls = 0, hash_calls = 0;
  auto ctx = in->GetContext();
  TableRowIndexComparator row_comp(ltab, cols);
  TableRowIndexHash row_hash(ltab, cols);
  auto buckets_pre_alloc = (ltab->num_rows());
  LOG(INFO) << "Buckets : " << buckets_pre_alloc;
  std::unordered_set<int64_t, TableRowIndexHash, TableRowIndexComparator>
      rows_set(buckets_pre_alloc, row_hash, row_comp);

  const int64_t num_rows = ltab->num_rows();
  //const int64_t print_threshold = num_rows / 10;
  auto p2 = std::chrono::high_resolution_clock::now();

  if (first) {
    for (int64_t row = 0; row < num_rows; ++row) {
      rows_set.insert(row);
//      if (row % print_threshold == 0) {
//        LOG(INFO) << "Done " << (row + 1) * 100 / num_rows << "%" << " N : "
//                  << row << ", Eq : " << eq_calls << ", Hs : "
//                  << hash_calls;
//      }
    }
  } else {
    for (int64_t row = num_rows - 1; row > 0; --row) {
      rows_set.insert(row);
//      if (row % print_threshold == 0) {
//        LOG(INFO) << "Done " << (row + 1) * 100 / num_rows << "%" << " N : "
//                  << row << ", Eq : " << eq_calls << ", Hs : "
//                  << hash_calls;
//      }
    }
  }
  auto p3 = std::chrono::high_resolution_clock::now();

//  std::shared_ptr<std::vector<int64_t>>
//      indices_from_tab = std::make_shared<std::vector<int64_t>>(rows_set.size());
//  std::copy(rows_set.begin(), rows_set.end(), indices_from_tab->begin());

  auto p4 = std::chrono::high_resolution_clock::now();

  std::vector<std::shared_ptr<arrow::ChunkedArray>> final_data_arrays;
  // prepare final arrays
  for (int32_t col_idx = 0; col_idx < ltab->num_columns(); col_idx++) {
    arrow::ArrayVector array_vector;
    Status status = prepare_array(ctx,
                                  ltab,
                                  col_idx,
                                  rows_set.begin(),
                                  rows_set.end(),
                                  rows_set.size(),
                                  array_vector);
    if (!status.is_ok()) return status;
    final_data_arrays.push_back(std::make_shared<arrow::ChunkedArray>(array_vector));
  }
  // create final table
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(ltab->schema(), final_data_arrays);
  arrow::Result<std::shared_ptr<arrow::Table>> merge_res = table->CombineChunks(cylon::ToArrowPool(ctx));
  if (!merge_res.ok()) {
    return Status(static_cast<int>(merge_res.status().code()), merge_res.status().message());
  }
  out = std::make_shared<cylon::Table>(merge_res.ValueOrDie(), ctx);
  auto p5 = std::chrono::high_resolution_clock::now();

  LOG(INFO) << ">>>>>P1 " << std::chrono::duration_cast<std::chrono::milliseconds>(
      p2 - p1).count() << "[s]";

  LOG(INFO) << ">>>>>P2 " << std::chrono::duration_cast<std::chrono::milliseconds>(
      p3 - p2).count() << "[s]";

  LOG(INFO) << ">>>>>P3 " << std::chrono::duration_cast<std::chrono::milliseconds>(
      p4 - p3).count() << "[s]";

  LOG(INFO) << ">>>>>P4 " << std::chrono::duration_cast<std::chrono::milliseconds>(
      p5 - p4).count() << "[s]";

  return Status::OK();
}

Status DistributedUnique(std::shared_ptr<cylon::Table> &in,
                         const std::vector<int> &cols,
                         std::shared_ptr<cylon::Table> &out) {
  auto ctx = in->GetContext();
  if (ctx->GetWorldSize() == 1) {
    return Unique(in, cols, out);
  }

  std::shared_ptr<cylon::Table> shuffle_out;
  auto shuffle_status = cylon::Shuffle(in, cols, shuffle_out);
  if (!shuffle_status.is_ok()) {
    LOG(ERROR) << "Shuffle Failed!";
    return shuffle_status;
  }

  return Unique(shuffle_out, cols, out);
}

#ifdef BUILD_CYLON_PARQUET
Status FromParquet(std::shared_ptr<cylon::CylonContext> &ctx,
                   const std::string &path,
                   std::shared_ptr<Table> &tableOut) {
  arrow::Result<std::shared_ptr<arrow::Table>> result = cylon::io::ReadParquet(ctx, path);
  if (result.ok()) {
    std::shared_ptr<arrow::Table> table = result.ValueOrDie();
    LOG(INFO) << "Chunks " << table->column(0)->chunks().size();
    if (table->column(0)->chunks().size() > 1) {
      auto combine_res = table->CombineChunks(ToArrowPool(ctx));
      if (!combine_res.ok()) {
        return Status(static_cast<int>(combine_res.status().code()), combine_res.status().message());
      }
      tableOut = std::make_shared<Table>(combine_res.ValueOrDie(), ctx);
    } else {
      tableOut = std::make_shared<Table>(table, ctx);
    }
    return Status::OK();
  }
  return Status(Code::IOError, result.status().message());
}

void ReadParquetThread(const std::shared_ptr<CylonContext> &ctx,
                       const std::string &path,
                       std::shared_ptr<cylon::Table> *table,
                       const std::shared_ptr<std::promise<Status>> &status_promise) {
  std::shared_ptr<CylonContext> ctx_ = ctx; // make a copy of the shared ptr
  status_promise->set_value(FromParquet(ctx_,
                                        path,
                                        *table));
}

Status FromParquet(std::shared_ptr<cylon::CylonContext> &ctx,
                   const std::vector<std::string> &paths,
                   const std::vector<std::shared_ptr<Table> *> &tableOuts,
                   io::config::ParquetOptions options) {
  if (options.IsConcurrentFileReads()) {
    std::vector<std::pair<std::future<Status>, std::thread>> futures;
    futures.reserve(paths.size());
    for (uint64_t kI = 0; kI < paths.size(); ++kI) {
      auto read_promise = std::make_shared<std::promise<Status>>();
      futures.emplace_back(read_promise->get_future(),
                           std::thread(
                               ReadParquetThread,
                               ctx,
                               paths[kI],
                               tableOuts[kI],
                               read_promise)
      );
    }
    bool all_passed = true;
    for (auto &future : futures) {
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

Status WriteParquet(std::shared_ptr<cylon::Table> &table,
                    std::shared_ptr<cylon::CylonContext> &ctx_,
                    const std::string &path,
                    const cylon::io::config::ParquetOptions &options) {
  arrow::Status writefile_result = cylon::io::WriteParquet(ctx_,
                                                           table,
                                                           path,
                                                           options);
  if (!writefile_result.ok()) {
    return Status(Code::IOError, writefile_result.message());
  }

  return Status(Code::OK);
}
#endif
}  // namespace cylon
