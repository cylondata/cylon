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

#ifndef GCYLON_TEST_UTILS_HPP_
#define GCYLON_TEST_UTILS_HPP_

#include <glog/logging.h>
#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/types.hpp>
#include <cudf/copying.hpp>

#include <gcylon/gtable_api.hpp>
#include <gcylon/utils/util.hpp>
#include <gcylon/net/cudf_net_ops.hpp>
#include <cudf/concatenate.hpp>

// this is a toggle to generate test files. Set execute to 0 then, it will generate the expected
// output files
#define EXECUTE 1

using namespace gcylon;

namespace gcylon {
namespace test {

cudf::io::table_with_metadata readCSV(const std::string &filename,
                                      const std::vector<std::string> &column_names = std::vector<std::string>{},
                                      const std::vector<std::string> &date_columns = std::vector<std::string>{}) {
  cudf::io::source_info si(filename);
  cudf::io::csv_reader_options options = cudf::io::csv_reader_options::builder(si);
  if (column_names.size() > 0) {
    options.set_use_cols_names(column_names);
  }
  if (date_columns.size() > 0) {
    options.set_infer_date_names(date_columns);
  }
  cudf::io::table_with_metadata ctable = cudf::io::read_csv(options);
  return ctable;
}

void writeCSV(cudf::table_view &tv, std::string filename, int rank, cudf::io::table_metadata &table_metadata) {
  cudf::io::sink_info sinkInfo(filename);
  cudf::io::csv_writer_options writer_options =
    cudf::io::csv_writer_options::builder(sinkInfo, tv).metadata(&table_metadata);
  LOG(INFO) << "myRank: " << rank << ", output file: " << filename
            << ", cols: " << tv.num_columns() << ", rows: " << tv.num_rows();
  cudf::io::write_csv(writer_options);
}

bool PerformShuffleTest(std::string &input_filename, std::string &output_filename, int shuffle_index) {
  std::vector<std::string> column_names{"city", "state_id", "population"};
  cudf::io::table_with_metadata input_table = readCSV(input_filename, column_names);
  auto input_tv = input_table.tbl->view();

  // shuffle the table
  std::vector<cudf::size_type> columns_to_hash = {shuffle_index};
  std::unique_ptr<cudf::table> shuffled_table;
  Shuffle(input_tv, columns_to_hash, ctx, shuffled_table);
  auto shuffled_tv = shuffled_table->view();

#if EXECUTE
  cudf::io::table_with_metadata saved_shuffled_table = readCSV(output_filename, column_names);
  auto saved_tv = saved_shuffled_table.tbl->view();
  return table_equal_with_sorting(shuffled_tv, saved_tv);
#else
  writeCSV(shuffled_tv, output_filename, rank, input_table.metadata);
  return true;
#endif
}

std::vector<std::string> constructInputFiles(const std::string &base, int world_size) {
  std::vector<std::string> all_input_files;

  for (int i = 0; i < world_size; i++) {
    std::string filename = base + std::to_string(i) + ".csv";
    all_input_files.push_back(filename);
  }
  return all_input_files;
}

/**
 * generate "count" random non-zero numbers totalling "max"
 * max has to be larger than count
 * @param max
 * @param count
 * @return
 */
std::vector<int32_t> GenRandoms(int max, int count, int seed) {
  srand (seed);
  std::vector<int32_t> randoms;
  randoms.reserve(count);

  for (int i = 0; i < count - 1; ++i) {
    int num = std::rand() % max;
    randoms.push_back(num);
    max -= num;
  }
  randoms.push_back(max);
  return randoms;
}

std::vector<std::unique_ptr<cudf::table>>
readTables(const std::string &input_file_base,
           const std::vector<std::string> &column_names,
           const std::vector<std::string> &date_columns) {

  std::vector<std::string> all_input_files = gcylon::test::constructInputFiles(input_file_base, WORLD_SZ);

  std::vector<std::unique_ptr<cudf::table>> tables;
  tables.reserve(all_input_files.size());
  for (long unsigned int i = 0; i < all_input_files.size(); i++) {
    cudf::io::table_with_metadata read_table = readCSV(all_input_files[i], column_names, date_columns);
    tables.push_back(std::move(read_table.tbl));
  }

  return tables;
}


std::unique_ptr<cudf::table>
concatSlices(const std::vector<std::unique_ptr<cudf::table>> &tables,
             const std::vector<std::vector<int32_t>> &ranges) {

  std::vector<cudf::table_view> views;
  views.reserve(tables.size());
  for (long unsigned int i = 0; i < tables.size(); i++) {
    auto slice_tv = cudf::slice(tables[i]->view(), ranges[i])[0];
    views.push_back(slice_tv);
  }

  return cudf::concatenate(views);
}

bool PerformGatherTest(const std::string &input_filename,
                       std::vector<std::string> &all_input_files,
                       const std::vector<std::string> &column_names,
                       const std::vector<std::string> &date_columns,
                       int gather_root,
                       bool gather_from_root,
                       std::shared_ptr<cylon::CylonContext> ctx) {

  cudf::io::table_with_metadata input_table = readCSV(input_filename, column_names, date_columns);
  auto input_tv = input_table.tbl->view();

  // gather the tables
  std::vector<std::unique_ptr<cudf::table>> gathered_tables;
  cylon::Status status = gcylon::net::Gather(input_tv,
                                             gather_root,
                                             gather_from_root,
                                             ctx,
                                             gathered_tables);
  if (!status.is_ok()) {
    return false;
  }

  // if not the root worker, nothing more to be done
  if (gather_root != ctx->GetRank()) {
    return true;
  }

  auto gathered_views = tablesToViews(gathered_tables);
  auto gathered_table = cudf::concatenate(gathered_views);

  std::vector<std::unique_ptr<cudf::table>> file_tables;
  file_tables.reserve(all_input_files.size());
  for (long unsigned int i = 0; i < all_input_files.size(); i++) {
    cudf::io::table_with_metadata read_table = readCSV(all_input_files[i], column_names, date_columns);
    file_tables.push_back(std::move(read_table.tbl));
  }
  auto file_views = tablesToViews(file_tables);
  auto file_table = cudf::concatenate(file_views);
  if (!table_equal(gathered_table->view(), file_table->view())) {
    return false;
  }

  return true;
}

bool PerformGatherSlicedTest(const cudf::table_view &input_tv,
                             const std::vector<std::string> &all_input_files,
                             const std::vector<std::string> &column_names,
                             const std::vector<std::string> &date_columns,
                             const std::vector<int32_t> &row_range,
                             std::shared_ptr<cylon::CylonContext> ctx) {

  int GATHER_ROOT = 0;
  bool GATHER_FROM_ROOT = true;

  // gather the tables
  std::vector<std::unique_ptr<cudf::table>> gathered_tables;
  cylon::Status status = gcylon::net::Gather(input_tv,
                                             GATHER_ROOT,
                                             GATHER_FROM_ROOT,
                                             ctx,
                                             gathered_tables);
  if (!status.is_ok()) {
    return false;
  }

  // if not the root worker, nothing more to be done
  if (GATHER_ROOT != ctx->GetRank()) {
    return true;
  }

  auto gathered_views = tablesToViews(gathered_tables);
  auto gathered_table = cudf::concatenate(gathered_views);

  std::vector<std::unique_ptr<cudf::table>> file_tables;
  file_tables.reserve(all_input_files.size());
  std::vector<cudf::table_view> file_views;
  file_views.reserve(all_input_files.size());
  for (long unsigned int i = 0; i < all_input_files.size(); i++) {
    cudf::io::table_with_metadata read_table = readCSV(all_input_files[i], column_names, date_columns);
    auto read_tv = cudf::slice(read_table.tbl->view(), row_range)[0];
    file_views.push_back(read_tv);
    file_tables.push_back(std::move(read_table.tbl));
  }

  auto file_table = cudf::concatenate(file_views);

  if (!table_equal(gathered_table->view(), file_table->view())) {
    return false;
  }
  return true;
}


bool PerformBcastTest(const cudf::table_view &input_tv,
                      int bcast_root,
                      std::shared_ptr<cylon::CylonContext> ctx) {

  // broadcast the table from broadcast root
  cudf::table_view send_tv;
  if (bcast_root == ctx->GetRank()) {
    send_tv = input_tv;
  }
  std::unique_ptr<cudf::table> received_table;
  cylon::Status status = gcylon::net::Bcast(send_tv, bcast_root, ctx, received_table);
  if (!status.is_ok()) {
    return false;
  }

  // compare received table to read table for all receiving workers
  if (bcast_root != ctx->GetRank()) {
    if (received_table == nullptr) {
      return false;
    }

    auto original_table = std::make_unique<cudf::table>(input_tv);
    if (!table_equal(original_table->view(), received_table->view())) {
      return false;
    }
  }

  return true;
}

bool PerformSortTest(const std::string &input_filename,
                     const std::string &sorted_filename,
                     const std::vector<std::string> &column_names,
                     const std::vector<std::string> &date_columns,
                     int sort_root,
                     const std::vector<int32_t> sort_columns,
                     std::shared_ptr<cylon::CylonContext> ctx) {

  cudf::io::table_with_metadata input_table = readCSV(input_filename, column_names, date_columns);
  auto input_tv = input_table.tbl->view();
  cudf::io::table_with_metadata sorted_saved_table = readCSV(sorted_filename, column_names, date_columns);

  std::vector<cudf::order> column_orders(sort_columns.size(), cudf::order::ASCENDING);

  // perform distributed sort
  std::unique_ptr<cudf::table> sorted_table;
  cylon::Status status = DistributedSort(input_tv,
                                         sort_columns,
                                         column_orders,
                                         ctx,
                                         sorted_table,
                                         true,
                                         sort_root);
  if (!status.is_ok()) {
    return false;
  }

  auto sorted_columns_tv = sorted_table->select(sort_columns);
  auto sorted_saved_columns_tv = sorted_saved_table.tbl->select(sort_columns);

  // compare resulting sorted table with the sorted table from the file
  return table_equal(sorted_columns_tv, sorted_saved_columns_tv);
}

bool PerformSlicedSortTest(const std::string &input_filename,
                           const std::string &sorted_filename,
                           const std::vector<std::string> &column_names,
                           const std::vector<std::string> &date_columns,
                           const std::vector<int32_t> sort_columns,
                           const std::vector<int32_t> slice_range,
                           std::shared_ptr<cylon::CylonContext> ctx) {

  cudf::io::table_with_metadata input_table = readCSV(input_filename, column_names, date_columns);
  auto input_tv = cudf::slice(input_table.tbl->view(), slice_range)[0];

  cudf::io::table_with_metadata sorted_saved_table = readCSV(sorted_filename, column_names, date_columns);

  std::vector<cudf::order> column_orders(sort_columns.size(), cudf::order::ASCENDING);

  // perform distributed sort
  std::unique_ptr<cudf::table> sorted_table;
  cylon::Status status = DistributedSort(input_tv,
                                         sort_columns,
                                         column_orders,
                                         ctx,
                                         sorted_table,
                                         true);
  if (!status.is_ok()) {
    return false;
  }

  auto sorted_columns_tv = sorted_table->select(sort_columns);
  auto sorted_saved_columns_tv = sorted_saved_table.tbl->select(sort_columns);

  // compare resulting sorted table with the sorted table from the file
  return table_equal(sorted_columns_tv, sorted_saved_columns_tv);
}

bool PerformRepartitionTest(const std::string &input_filename,
                            const std::vector<std::string> &column_names,
                            const std::vector<std::string> &date_columns,
                            const std::shared_ptr<cylon::CylonContext> &ctx,
                            const std::vector<int32_t> &initial_sizes,
                            const std::vector<int32_t> &part_sizes = std::vector<int32_t>()) {

  cudf::io::table_with_metadata input_table = readCSV(input_filename, column_names, date_columns);
  std::vector<cudf::size_type> range {0, initial_sizes[ctx->GetRank()]};
  auto sub_tv = cudf::slice(input_table.tbl->view(), range)[0];

  // repartition the table
  std::unique_ptr<cudf::table> table_out;
  cylon::Status status = Repartition(sub_tv, ctx, table_out, part_sizes);
  if (!status.is_ok()) {
    return false;
  }

  // check the number of rows in the repartitioned table
  int32_t table_size = 0;
  if (part_sizes.empty()) {
    int32_t sum_of_rows = std::accumulate(initial_sizes.begin(), initial_sizes.end(), 0);
    table_size = sum_of_rows / 4;
  } else {
    table_size = part_sizes[ctx->GetRank()];
  }

  if (table_out->num_rows() != table_size) {
    return false;
  }

  // gather the initial tables to the first worker and compare two tables
  std::unique_ptr<cudf::table> gathered_init_table;
  status = gcylon::Gather(sub_tv, ctx, gathered_init_table);
  if (!status.is_ok()) {
    return false;
  }

  std::unique_ptr<cudf::table> gathered_repart_table;
  status = gcylon::Gather(table_out->view(), ctx, gathered_repart_table);
  if (!status.is_ok()) {
    return false;
  }

  if(!table_equal(gathered_init_table->view(), gathered_repart_table->view())) {
    return false;
  }

  return true;
}

bool PerformReplicateTest(const std::vector<std::unique_ptr<cudf::table>> &tables,
                          const std::vector<std::vector<int32_t>> &ranges,
                          const std::shared_ptr<cylon::CylonContext> &ctx) {

  auto tv_slice = cudf::slice(tables[RANK]->view(), ranges[RANK])[0];

  // allgather the tables
  std::unique_ptr<cudf::table> gathered_table;
  cylon::Status status = gcylon::Replicate(tv_slice, ctx,gathered_table);
  if (!status.is_ok()) {
    return false;
  }

  auto file_table = concatSlices(tables, ranges);

  if (!table_equal(gathered_table->view(), file_table->view())) {
    return false;
  }
  return true;
}


}
}
#endif //CYLON_CPP_SRC_EXAMPLES_TEST_UTILS_HPP_
