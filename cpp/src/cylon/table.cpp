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

#include "table_api_extended.hpp"
#include "io/arrow_io.hpp"
#include "join/join.hpp"
#include  "util/to_string.hpp"
#include "iostream"
#include "util/arrow_utils.hpp"
#include "arrow/arrow_partition_kernels.hpp"
#include "util/uuid.hpp"
#include "arrow/arrow_all_to_all.hpp"
#include "arrow/arrow_comparator.hpp"
#include "ctx/arrow_memory_pool_utils.hpp"
#include "arrow/arrow_types.hpp"

namespace cylon {

class RowComparator {
 private:
  const std::shared_ptr<arrow::Table> *tables;
  std::shared_ptr<cylon::TableRowComparator> comparator;
  std::shared_ptr<cylon::RowHashingKernel> row_hashing_kernel;
  int64_t *eq, *hs;

 public:
  RowComparator(std::shared_ptr<cylon::CylonContext> &ctx,
				const std::shared_ptr<arrow::Table> *tables,
				int64_t *eq,
				int64_t *hs) {
	this->tables = tables;
	this->comparator = std::make_shared<cylon::TableRowComparator>(tables[0]->fields());
	this->row_hashing_kernel = std::make_shared<cylon::RowHashingKernel>(tables[0]->fields(),
																		 cylon::ToArrowPool(ctx));
	this->eq = eq;
	this->hs = hs;
  }

  // equality
  bool operator()(const std::pair<int8_t, int64_t> &record1,
				  const std::pair<int8_t, int64_t> &record2) const {
	(*this->eq)++;
	return this->comparator->compare(this->tables[record1.first], record1.second,
									 this->tables[record2.first], record2.second) == 0;
  }

  // hashing
  size_t operator()(const std::pair<int8_t, int64_t> &record) const {
	(*this->hs)++;
	size_t hash = this->row_hashing_kernel->Hash(this->tables[record.first], record.second);
	return hash;
  }
};

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
  arrow::Status ar_status = cylon::util::copy_array_by_indices(row_indices,
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

Status HashPartitionTable(std::shared_ptr<cylon::CylonContext> &ctx, const std::shared_ptr<arrow::Table> &table,
						  int hash_column, int no_of_partitions,
						  std::unordered_map<int, std::shared_ptr<cylon::Table>> *out) {
  // keep arrays for each target, these arrays are used for creating the table
  std::unordered_map<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>> data_arrays;
  std::vector<int> partitions;
  for (int t = 0; t < no_of_partitions; t++) {
	partitions.push_back(t);
	data_arrays.insert(
		std::pair<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>>(
			t, std::make_shared<std::vector<std::shared_ptr<arrow::Array>>>()));
  }
  std::shared_ptr<arrow::Array> arr = table->column(hash_column)->chunk(0);
  int64_t length = arr->length();

  auto t1 = std::chrono::high_resolution_clock::now();
  // first we partition the table
  std::vector<int64_t> outPartitions;
  outPartitions.reserve(length);
  std::vector<uint32_t> counts(no_of_partitions, 0);
  Status status = HashPartitionArray(cylon::ToArrowPool(ctx), arr,
									 partitions, &outPartitions, counts);
  if (!status.is_ok()) {
	LOG(FATAL) << "Failed to create the hash partition";
	return status;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Calculating hash time : "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  for (int i = 0; i < table->num_columns(); i++) {
	std::shared_ptr<arrow::DataType> type = table->column(i)->chunk(0)->type();
	std::shared_ptr<arrow::Array> array = table->column(i)->chunk(0);

	std::shared_ptr<ArrowArraySplitKernel> splitKernel;
	status = CreateSplitter(type, cylon::ToArrowPool(ctx), &splitKernel);
	if (!status.is_ok()) {
	  LOG(FATAL) << "Failed to create the splitter";
	  return status;
	}
	// this one outputs arrays for each target as a map
	std::unordered_map<int, std::shared_ptr<arrow::Array>> splited_arrays;
	splitKernel->Split(array, outPartitions, partitions, splited_arrays, counts);
	for (const auto &x : splited_arrays) {
	  std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>> cols = data_arrays[x.first];
	  cols->push_back(x.second);
	}
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Building hashed tables time : "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
  // now insert these array to
  for (const auto &x : data_arrays) {
	std::shared_ptr<arrow::Table> t = arrow::Table::Make(table->schema(), *x.second);
	std::shared_ptr<cylon::Table> kY = std::make_shared<cylon::Table>(t, ctx);
	out->insert(std::pair<int, std::shared_ptr<cylon::Table>>(x.first, kY));
  }
  return Status::OK();
}

cylon::Status Shuffle(std::shared_ptr<cylon::CylonContext> &ctx,
					  std::shared_ptr<cylon::Table> &table,
					  int hash_column,
					  int edge_id,
					  std::shared_ptr<arrow::Table> *table_out) {
  std::unordered_map<int, std::shared_ptr<cylon::Table>> partitioned_tables{};
  // partition the tables locally
  HashPartitionTable(ctx, table->get_table(), hash_column, ctx->GetWorldSize(),
					 &partitioned_tables);
  std::shared_ptr<arrow::Schema> schema = table->get_table()->schema();
  // we are going to free if retain is set to false
  if (!table->IsRetain()) {
	table.reset();
  }
  auto neighbours = ctx->GetNeighbours(true);
  vector<std::shared_ptr<arrow::Table>> received_tables;
  // define call back to catch the receiving tables
  class AllToAllListener : public cylon::ArrowCallback {
	vector<std::shared_ptr<arrow::Table>> *tabs;
	int workerId;

   public:
	explicit AllToAllListener(vector<std::shared_ptr<arrow::Table>> *tabs, int workerId) {
	  this->tabs = tabs;
	  this->workerId = workerId;
	}

	bool onReceive(int source, std::shared_ptr<arrow::Table> table) override {
	  this->tabs->push_back(table);
	  return true;
	};
  };

  // doing all to all communication to exchange tables
  cylon::ArrowAllToAll all_to_all(ctx, neighbours, neighbours, edge_id,
								  std::make_shared<AllToAllListener>(&received_tables,
																	 ctx->GetRank()), schema, cylon::ToArrowPool(ctx));

  for (auto &partitioned_table : partitioned_tables) {
	if (partitioned_table.first != ctx->GetRank()) {
	  all_to_all.insert(partitioned_table.second->get_table(), partitioned_table.first);
	} else {
	  received_tables.push_back(partitioned_table.second->get_table());
	}
  }

  // now complete the communication
  all_to_all.finish();
  while (!all_to_all.isComplete()) {}
  all_to_all.close();

  // now clear locally partitioned tables
  partitioned_tables.clear();

  // now we have the final set of tables
  LOG(INFO) << "Concatenating tables, Num of tables :  " << received_tables.size();
  arrow::Result<std::shared_ptr<arrow::Table>> concat_tables =
	  arrow::ConcatenateTables(received_tables);

  if (concat_tables.ok()) {
	auto final_table = concat_tables.ValueOrDie();
	LOG(INFO) << "Done concatenating tables, rows :  " << final_table->num_rows();
	auto status = final_table->CombineChunks(cylon::ToArrowPool(ctx), table_out);
	return Status(static_cast<int>(status.code()), status.message());
  } else {
	return Status(static_cast<int>(concat_tables.status().code()),
				  concat_tables.status().message());
  }
}

cylon::Status Shuffle(std::shared_ptr<cylon::CylonContext> &ctx,
					  std::shared_ptr<cylon::Table> &table,
					  const std::vector<int> &hash_columns,
					  int edge_id,
					  std::shared_ptr<arrow::Table> *table_out) {
  std::unordered_map<int, std::shared_ptr<cylon::Table>> partitioned_tables{};
  // partition the tables locally
  table->HashPartition(hash_columns, ctx->GetWorldSize(), &partitioned_tables);
  std::shared_ptr<arrow::Schema> schema = table->get_table()->schema();
  // we are going to free if retain is set to false
  if (!table->IsRetain()) {
	table.reset();
  }
  auto neighbours = ctx->GetNeighbours(true);
  vector<std::shared_ptr<arrow::Table>> received_tables;
  // define call back to catch the receiving tables
  class AllToAllListener : public cylon::ArrowCallback {
	vector<std::shared_ptr<arrow::Table>> *tabs;
	int workerId;

   public:
	explicit AllToAllListener(vector<std::shared_ptr<arrow::Table>> *tabs, int workerId) {
	  this->tabs = tabs;
	  this->workerId = workerId;
	}

	bool onReceive(int source, std::shared_ptr<arrow::Table> table) override {
	  this->tabs->push_back(table);
	  return true;
	};
  };

  // doing all to all communication to exchange tables
  cylon::ArrowAllToAll all_to_all(ctx, neighbours, neighbours, edge_id,
								  std::make_shared<AllToAllListener>(&received_tables,
																	 ctx->GetRank()), schema, cylon::ToArrowPool(ctx));

  for (auto &partitioned_table : partitioned_tables) {
	if (partitioned_table.first != ctx->GetRank()) {
	  all_to_all.insert(partitioned_table.second->get_table(), partitioned_table.first);
	} else {
	  received_tables.push_back(partitioned_table.second->get_table());
	}
  }

  // now complete the communication
  all_to_all.finish();
  while (!all_to_all.isComplete()) {}
  all_to_all.close();

  // now clear locally partitioned tables
  partitioned_tables.clear();

  // now we have the final set of tables
  LOG(INFO) << "Concatenating tables, Num of tables :  " << received_tables.size();
  arrow::Result<std::shared_ptr<arrow::Table>> concat_tables =
	  arrow::ConcatenateTables(received_tables);

  if (concat_tables.ok()) {
	auto final_table = concat_tables.ValueOrDie();
	LOG(INFO) << "Done concatenating tables, rows :  " << final_table->num_rows();
	auto status = final_table->CombineChunks(cylon::ToArrowPool(ctx), table_out);
	return Status(static_cast<int>(status.code()), status.message());
  } else {
	return Status(static_cast<int>(concat_tables.status().code()),
				  concat_tables.status().message());
  }
}

Status ShuffleTwoTables(std::shared_ptr<cylon::CylonContext> &ctx,
						std::shared_ptr<cylon::Table> &left_table,
						int left_hash_column,
						std::shared_ptr<cylon::Table> &right_table,
						int right_hash_column,
						std::shared_ptr<arrow::Table> *left_table_out,
						std::shared_ptr<arrow::Table> *right_table_out) {
  LOG(INFO) << "Shuffling two tables with total rows : "
			<< left_table->Rows() + right_table->Rows();
  auto t1 = std::chrono::high_resolution_clock::now();
  auto status = Shuffle(ctx, left_table, left_hash_column,
						ctx->GetNextSequence(), left_table_out);
  if (status.is_ok()) {
	auto t2 = std::chrono::high_resolution_clock::now();
	LOG(INFO) << "Left shuffle time : "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	status = Shuffle(ctx, right_table, right_hash_column,
					 ctx->GetNextSequence(), right_table_out);
	if (status.is_ok()) {
	  auto t3 = std::chrono::high_resolution_clock::now();
	  LOG(INFO) << "Right shuffle time : "
				<< std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
	}
  }
  return status;
}

Status ShuffleTwoTables(std::shared_ptr<cylon::CylonContext> &ctx,
						std::shared_ptr<cylon::Table> &left_table,
						const std::vector<int> &left_hash_columns,
						std::shared_ptr<cylon::Table> &right_table,
						const std::vector<int> &right_hash_columns,
						std::shared_ptr<arrow::Table> *left_table_out,
						std::shared_ptr<arrow::Table> *right_table_out) {
  LOG(INFO) << "Shuffling two tables with total rows : "
			<< left_table->Rows() + right_table->Rows();
  auto t1 = std::chrono::high_resolution_clock::now();
  auto status = Shuffle(ctx, left_table, left_hash_columns,
						ctx->GetNextSequence(), left_table_out);
  if (status.is_ok()) {
	auto t2 = std::chrono::high_resolution_clock::now();
	LOG(INFO) << "Left shuffle time : "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	status = Shuffle(ctx, right_table, right_hash_columns,
					 ctx->GetNextSequence(), right_table_out);
	if (status.is_ok()) {
	  auto t3 = std::chrono::high_resolution_clock::now();
	  LOG(INFO) << "Right shuffle time : "
				<< std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
	}
  }
  return status;
}

Status Table::FromCSV(std::shared_ptr<cylon::CylonContext> &ctx, const std::string &path,
					  std::shared_ptr<Table> &tableOut,
					  const cylon::io::config::CSVReadOptions &options) {
  arrow::Result<std::shared_ptr<arrow::Table>> result = cylon::io::read_csv(ctx, path, options);
  if (result.ok()) {
	std::shared_ptr<arrow::Table> table = *result;
	LOG(INFO) << "Chunks " << table->column(0)->chunks().size();
	if (table->column(0)->chunks().size() > 1) {
	  auto status = table->CombineChunks(ToArrowPool(ctx), &table);
	  if (!status.ok()) {
		return Status(Code::IOError, status.message());
	  }
	}
	tableOut = std::make_shared<Table>(table, ctx);
	return Status(Code::OK, result.status().message());
  }
  return Status(Code::IOError, result.status().message());
}

Status Table::FromArrowTable(std::shared_ptr<cylon::CylonContext> &ctx,
							 std::shared_ptr<arrow::Table> &table,
							 std::shared_ptr<Table> *tableOut) {
  if (!cylon::tarrow::validateArrowTableTypes(table)) {
	LOG(FATAL) << "Types not supported";
	return Status(cylon::Invalid, "This type not supported");
  }
  *tableOut = std::make_shared<Table>(table, ctx);
  return Status(cylon::OK, "Loaded Successfully");
}

Status Table::FromColumns(std::shared_ptr<cylon::CylonContext> &ctx,
						  vector<std::shared_ptr<Column>> &&columns,
						  std::shared_ptr<Table> *tableOut) {
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
  shared_ptr<arrow::Table> arrow_table = arrow::Table::Make(schema_result.ValueOrDie(), col_arrays);

  if (!cylon::tarrow::validateArrowTableTypes(arrow_table)) {
	LOG(FATAL) << "Types not supported";
	return Status(cylon::Invalid, "This type not supported");
  }
  *tableOut = std::make_shared<Table>(arrow_table, ctx, columns);

  return Status(cylon::OK, "Loaded Successfully");
}

Status Table::WriteCSV(const std::string &path, const cylon::io::config::CSVWriteOptions &options) {
  std::ofstream out_csv;
  out_csv.open(path);
  Status status = PrintToOStream(0,
								 table_->num_columns(), 0,
								 table_->num_rows(), out_csv,
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

Status Table::Merge(std::shared_ptr<cylon::CylonContext> &ctx,
					const std::vector<std::shared_ptr<cylon::Table>> &ctables,
					shared_ptr<Table> &tableOut) {
  std::vector<std::shared_ptr<arrow::Table>> tables;
  for (auto it = ctables.begin(); it < ctables.end(); it++) {
	std::shared_ptr<arrow::Table> arrow;
	(*it)->ToArrowTable(arrow);
	tables.push_back(arrow);
  }
  arrow::Result<std::shared_ptr<arrow::Table>> result = arrow::ConcatenateTables(tables);
  if (result.status() == arrow::Status::OK()) {
	std::shared_ptr<arrow::Table> combined;
	arrow::Status status = result.ValueOrDie()->CombineChunks(cylon::ToArrowPool(ctx), &combined);
	if (status != arrow::Status::OK()) {
	  return Status(Code::OutOfMemory);
	}
	tableOut = std::make_shared<cylon::Table>(combined, ctx);
	return Status::OK();
  } else {
	return Status(static_cast<int>(result.status().code()), result.status().message());
  }
}

Status Table::Sort(int sort_column, shared_ptr<cylon::Table> &out) {
  auto col = table_->column(sort_column)->chunk(0);
  std::shared_ptr<arrow::Array> indexSorts;
  arrow::Status status = SortIndices(cylon::ToArrowPool(ctx), col, &indexSorts);

  if (status != arrow::Status::OK()) {
	LOG(FATAL) << "Failed when sorting table to indices. " << status.ToString();
	return Status(static_cast<int>(status.code()), status.message());
  }

  std::vector<std::shared_ptr<arrow::Array>> data_arrays;
  for (auto &column : table_->columns()) {
	std::shared_ptr<arrow::Array> destination_col_array;
	status = cylon::util::copy_array_by_indices(nullptr, column->chunk(0),
												&destination_col_array, cylon::ToArrowPool(ctx));
	if (status != arrow::Status::OK()) {
	  LOG(FATAL) << "Failed while copying a column to the final table from left table. "
				 << status.ToString();
	  return Status(static_cast<int>(status.code()), status.message());
	}
	data_arrays.push_back(destination_col_array);
  }
  // we need to put this to a new place
  std::shared_ptr<arrow::Table> sortedTable = arrow::Table::Make(table_->schema(), data_arrays);
  return Status::OK();
}

Status Table::HashPartition(const std::vector<int> &hash_columns, int no_of_partitions,
							std::unordered_map<int, std::shared_ptr<cylon::Table>> *out) {
  // keep arrays for each target, these arrays are used for creating the table
  std::unordered_map<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>> data_arrays;
  std::vector<int> partitions;
  for (int t = 0; t < no_of_partitions; t++) {
	partitions.push_back(t);
	data_arrays.insert(
		std::pair<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>>(
			t, std::make_shared<std::vector<std::shared_ptr<arrow::Array>>>()));
  }

  std::vector<std::shared_ptr<arrow::Array>> arrays;
  int64_t length = 0;
  for (auto col_index : hash_columns) {
	auto column = table_->column(col_index);
	std::shared_ptr<arrow::Array> array = column->chunk(0);
	arrays.push_back(array);

	if (!(length == 0 || length == column->length())) {
	  return Status(cylon::IndexError,
					"Column lengths doesnt match " + std::to_string(length));
	}
	length = column->length();
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  // first we partition the table
  std::vector<int64_t> outPartitions;
  outPartitions.reserve(length);
  std::vector<uint32_t> counts(no_of_partitions, 0);
  Status status = HashPartitionArrays(cylon::ToArrowPool(ctx), arrays, length,
									  partitions, &outPartitions, counts);
  if (!status.is_ok()) {
	LOG(FATAL) << "Failed to create the hash partition";
	return status;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Calculating hash time : "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

  for (int i = 0; i < table_->num_columns(); i++) {
	std::shared_ptr<arrow::DataType> type = table_->column(i)->chunk(0)->type();
	std::shared_ptr<arrow::Array> array = table_->column(i)->chunk(0);

	std::shared_ptr<ArrowArraySplitKernel> splitKernel;
	status = CreateSplitter(type, cylon::ToArrowPool(ctx), &splitKernel);
	if (!status.is_ok()) {
	  LOG(FATAL) << "Failed to create the splitter";
	  return status;
	}

	// this one outputs arrays for each target as a map
	std::unordered_map<int, std::shared_ptr<arrow::Array>> splited_arrays;
	splitKernel->Split(array, outPartitions, partitions, splited_arrays, counts);

	for (const auto &x : splited_arrays) {
	  std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>> cols = data_arrays[x.first];
	  cols->push_back(x.second);
	}
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Building hashed tables time : "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
  // now insert these array to
  for (const auto &x : data_arrays) {
	std::shared_ptr<arrow::Table> table = arrow::Table::Make(table_->schema(), *x.second);
	std::shared_ptr<cylon::Table> kY = std::make_shared<cylon::Table>(table, this->ctx);
	out->insert(std::pair<int, std::shared_ptr<cylon::Table>>(x.first, kY));
  }
  return Status::OK();
}

Status Table::Join(std::shared_ptr<cylon::Table> &left, std::shared_ptr<cylon::Table> &right,
				   cylon::join::config::JoinConfig join_config,
				   std::shared_ptr<cylon::Table> *out) {
  if (left == NULLPTR) {
	return Status(Code::KeyError, "Couldn't find the left table");
  } else if (right == NULLPTR) {
	return Status(Code::KeyError, "Couldn't find the right table");
  } else {
	std::shared_ptr<arrow::Table> table;
	std::shared_ptr<arrow::Table> left_table;
	std::shared_ptr<arrow::Table> right_table;

	left->ToArrowTable(left_table);
	right->ToArrowTable(right_table);
	arrow::Status status = join::joinTables(
		left_table,
		right_table,
		join_config,
		&table,
		cylon::ToArrowPool(left->ctx));
	if (status == arrow::Status::OK()) {
	  *out = std::make_shared<cylon::Table>(table, left->ctx);
	}
	return Status(static_cast<int>(status.code()), status.message());
  }
}

Status Table::ToArrowTable(std::shared_ptr<arrow::Table> &out) {
  out = table_;
  return Status::OK();
}

Status Table::DistributedJoin(std::shared_ptr<cylon::Table> &left,
							  std::shared_ptr<cylon::Table> &right,
							  cylon::join::config::JoinConfig join_config,
							  std::shared_ptr<cylon::Table> *out) {
  // check whether the world size is 1
  shared_ptr<cylon::CylonContext> ctx = left->ctx;
  if (ctx->GetWorldSize() == 1) {
	cylon::Status status = Table::Join(
		left,
		right,
		join_config,
		out);
	return status;
  }

  std::vector<int> right_hash_columns;
  right_hash_columns.push_back(join_config.GetRightColumnIdx());
  std::shared_ptr<arrow::Table> left_final_table;
  std::shared_ptr<arrow::Table> right_final_table;
  auto shuffle_status = ShuffleTwoTables(ctx,
										 left,
										 join_config.GetLeftColumnIdx(),
										 right,
										 join_config.GetRightColumnIdx(),
										 &left_final_table,
										 &right_final_table);
  if (shuffle_status.is_ok()) {
	// now do the local join
	std::shared_ptr<arrow::Table> table;
	arrow::Status status = join::joinTables(
		left_final_table,
		right_final_table,
		join_config,
		&table,
		cylon::ToArrowPool(ctx));
	*out = std::make_shared<cylon::Table>(table, ctx);
	return Status(static_cast<int>(status.code()), status.message());
  } else {
	return shuffle_status;
  }
}

Status Table::Select(const std::function<bool(cylon::Row)> &selector, shared_ptr<Table> &out) {
  // boolean builder to hold the mask
  arrow::BooleanBuilder boolean_builder(cylon::ToArrowPool(ctx));
  for (int64_t row_index = 0; row_index < Rows(); row_index++) {
	auto row = cylon::Row(table_, row_index);
	arrow::Status status = boolean_builder.Append(selector(row));
	if (!status.ok()) {
	  return Status(UnknownError, status.message());
	}
  }
  // building the mask
  std::shared_ptr<arrow::Array> mask;
  arrow::Status status = boolean_builder.Finish(&mask);
  if (!status.ok()) {
	return Status(UnknownError, status.message());
  }
  std::shared_ptr<arrow::Table> out_table;
  arrow::compute::FunctionContext func_ctx;
  status = arrow::compute::Filter(&func_ctx, *table_, *mask, &out_table);
  if (!status.ok()) {
	out = std::make_shared<cylon::Table>(out_table, this->ctx);
	return Status(UnknownError, status.message());
  }
  return Status::OK();
}

Status Table::Union(std::shared_ptr<Table> &first, std::shared_ptr<Table> &second,
					std::shared_ptr<Table> &out) {
  shared_ptr<arrow::Table> ltab = first->get_table();
  shared_ptr<arrow::Table> rtab = second->get_table();
  Status status = VerifyTableSchema(ltab, second->get_table());
  if (!status.is_ok()) return status;
  std::shared_ptr<arrow::Table> tables[2] = {ltab, second->get_table()};
  int64_t eq_calls = 0, hash_calls = 0;

  auto row_comp = RowComparator(first->ctx, tables, &eq_calls, &hash_calls);
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
	  status = PrepareArray(first->ctx,
							tables[tab_idx],
							col_idx,
							indices_from_tabs[tab_idx],
							array_vector);

	  if (!status.is_ok()) return status;
	}
	final_data_arrays.push_back(std::make_shared<arrow::ChunkedArray>(array_vector));
  }
  // create final table
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(ltab->schema(), final_data_arrays);
  auto merge_status = table->CombineChunks(cylon::ToArrowPool(first->ctx), &table);
  if (!merge_status.ok()) {
	return Status(static_cast<int>(merge_status.code()), merge_status.message());
  }
  out = std::make_shared<cylon::Table>(table, first->ctx);
  return Status::OK();
}

Status Table::Subtract(shared_ptr<Table> &first,
					   shared_ptr<Table> &second, std::shared_ptr<Table> &out) {
  shared_ptr<arrow::Table> ltab = first->get_table();
  shared_ptr<arrow::Table> rtab = second->get_table();
  Status status = VerifyTableSchema(ltab, rtab);
  if (!status.is_ok()) {
	return status;
  }
  std::shared_ptr<arrow::Table> tables[2] = {ltab, rtab};
  int64_t eq_calls = 0, hash_calls = 0;
  auto row_comp = RowComparator(first->ctx, tables, &eq_calls, &hash_calls);
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
	status = PrepareArray(first->ctx, ltab, col_idx, left_indices, array_vector);
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
  out = std::make_shared<cylon::Table>(table, first->ctx);
  return Status::OK();
}

Status Table::Intersect(shared_ptr<Table> &first,
						shared_ptr<Table> &second, std::shared_ptr<Table> &out) {
  shared_ptr<arrow::Table> ltab = first->get_table();
  shared_ptr<arrow::Table> rtab = second->get_table();
  Status status = VerifyTableSchema(ltab, rtab);
  if (!status.is_ok()) {
	return status;
  }
  std::shared_ptr<arrow::Table> tables[2] = {ltab, rtab};
  int64_t eq_calls = 0, hash_calls = 0;
  auto row_comp = RowComparator(first->ctx, tables, &eq_calls, &hash_calls);
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
  std::shared_ptr<std::vector<int64_t>> left_indices = std::make_shared<std::vector<int64_t>>();;
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
	status = PrepareArray(first->ctx, ltab, col_idx, left_indices, array_vector);

	if (!status.is_ok()) return status;

	final_data_arrays.push_back(std::make_shared<arrow::ChunkedArray>(array_vector));
  }

  t2 = std::chrono::steady_clock::now();
  LOG(INFO) << "Final array preparation took "
			<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
			<< "ms";
  // create final table
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(ltab->schema(), final_data_arrays);
  out = std::make_shared<cylon::Table>(table, first->ctx);
  return Status::OK();
}

typedef Status
(*LocalSetOperation)(std::shared_ptr<cylon::Table> &, std::shared_ptr<cylon::Table> &,
					 std::shared_ptr<cylon::Table> &);

Status DoDistributedSetOperation(std::shared_ptr<cylon::CylonContext> &ctx,
								 LocalSetOperation local_operation,
								 std::shared_ptr<cylon::Table> &table_left,
								 std::shared_ptr<cylon::Table> &table_right,
								 std::shared_ptr<cylon::Table> &out) {
  // extract the tables out
  auto left = table_left->get_table();
  auto right = table_right->get_table();

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
  auto shuffle_status = ShuffleTwoTables(ctx,
										 table_left,
										 hash_columns,
										 table_right,
										 hash_columns,
										 &left_final_table,
										 &right_final_table);
  if (shuffle_status.is_ok()) {
	std::shared_ptr<cylon::Table> left_tab = std::make_shared<cylon::Table>(left_final_table, ctx);
	std::shared_ptr<cylon::Table> right_tab = std::make_shared<cylon::Table>(
		right_final_table, ctx);
	// now do the local union
	std::shared_ptr<arrow::Table> table;
	status = local_operation(left_tab, right_tab, out);
	return status;
  } else {
	return shuffle_status;
  }
}

Status Table::DistributedUnion(shared_ptr<Table> &left, shared_ptr<Table> &right,
							   shared_ptr<Table> &out) {
  return DoDistributedSetOperation(left->ctx, &Table::Union, left, right, out);
}

Status Table::DistributedSubtract(shared_ptr<Table> &left, shared_ptr<Table> &right,
								  shared_ptr<Table> &out) {
  return DoDistributedSetOperation(left->ctx, &Table::Subtract, left, right, out);
}

Status Table::DistributedIntersect(shared_ptr<Table> &left, shared_ptr<Table> &right,
								   shared_ptr<Table> &out) {
  return DoDistributedSetOperation(left->ctx, &Table::Intersect, left, right, out);
}

void Table::Clear() {
}

Table::~Table() {
  this->Clear();
}

void ReadCSVThread(cylon::CylonContext *ctx, const std::string &path,
				   std::shared_ptr<cylon::Table> *table,
				   cylon::io::config::CSVReadOptions options,
				   const std::shared_ptr<std::promise<Status>> &status_promise) {
  status_promise->set_value(Table::FromCSV(reinterpret_cast<shared_ptr<cylon::CylonContext> &>(ctx),
										   path,
										   *table,
										   options));
}

Status Table::FromCSV(std::shared_ptr<cylon::CylonContext> &ctx, const vector<std::string> &paths,
					  const std::vector<std::shared_ptr<Table> *> &tableOuts,
					  io::config::CSVReadOptions options) {
  if (options.IsConcurrentFileReads()) {
	std::vector<std::pair<std::future<Status>, std::thread>> futures;
	futures.reserve(paths.size());
	for (uint64_t kI = 0; kI < paths.size(); ++kI) {
	  auto read_promise = std::make_shared<std::promise<Status>>();
	  auto context = ctx.get();
	  futures.emplace_back(read_promise->get_future(),
						   std::thread(ReadCSVThread,
									   context,
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
	  status = Table::FromCSV(ctx, paths[kI], *tableOuts[kI], options);
	  if (!status.is_ok()) {
		return status;
	  }
	}
	return status;
  }
}

Status Table::Project(const std::vector<int64_t> &project_columns, std::shared_ptr<Table> &out) {
  std::vector<std::shared_ptr<arrow::Field>> schema_vector;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> column_arrays;
  schema_vector.reserve(project_columns.size());

  for (auto const &col_index : project_columns) {
	schema_vector.push_back(table_->field(col_index));
	auto chunked_array = std::make_shared<arrow::ChunkedArray>(table_->column(col_index)->chunks());
	column_arrays.push_back(chunked_array);
  }

  auto schema = std::make_shared<arrow::Schema>(schema_vector);
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, column_arrays);
  out = std::make_shared<cylon::Table>(table, this->ctx);
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
	  if (headers.size() != (uint64_t)table->num_columns()) {
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

shared_ptr<arrow::Table> Table::get_table() {
  return table_;
}

bool Table::IsRetain() const {
  return retain_;
}

std::shared_ptr<Column> Table::GetColumn(int32_t index) const {
  return this->columns_.at(index);
}

std::vector<shared_ptr<cylon::Column>> Table::GetColumns() const {
  return this->columns_;
}
}  // namespace cylon
