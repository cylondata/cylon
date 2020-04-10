#include <mpi.h>
#include <iostream>

#include <arrow/api.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array.h>
#include <glog/logging.h>
#include <arrow/compute/api.h>
#include <chrono>
#include <ctime>
#include <util/arrow_utils.hpp>
#include "join/join.cpp"

#include "arrow/arrow_kernels.hpp"

using arrow::DoubleBuilder;
using arrow::Int64Builder;
using arrow::FloatBuilder;

void merge_test() {
  arrow::MemoryPool *pool = arrow::default_memory_pool();

  Int64Builder left_id_builder(pool);
  Int64Builder right_id_builder(pool);
  Int64Builder cost_builder(pool);

  srand(std::time(NULL));

  int count = 400;
  int range = count * 10;

  for (int i = 0; i < count; i++) {
	int l = rand() % range;
	int r = rand() % range;

	//LOG(INFO) << "adding " << r << "and" << l;
	left_id_builder.Append(l);
	right_id_builder.Append(r);
	cost_builder.Append(i);
  }

  LOG(INFO) << "added";

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
	  arrow::field("id", arrow::int64()), arrow::field("cost", arrow::int64())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  std::shared_ptr<arrow::Array> left_id_array;
  left_id_builder.Finish(&left_id_array);
  std::shared_ptr<arrow::Array> right_id_array;
  right_id_builder.Finish(&right_id_array);

  std::shared_ptr<arrow::Array> cost_array;
  cost_builder.Finish(&cost_array);

  std::shared_ptr<arrow::Table> left_table = arrow::Table::Make(schema, {left_id_array, cost_array});
  std::shared_ptr<arrow::Table> right_table = arrow::Table::Make(schema, {right_id_array, cost_array});
  std::shared_ptr<arrow::Table> joined_table;

  LOG(INFO) << "Starting join";
  auto start = std::chrono::high_resolution_clock::now();
  twisterx::join::joinTables(
      left_table,
      right_table,
      0,
      0,
      twisterx::join::JoinType::INNER,
      twisterx::join::JoinAlgorithm::SORT,
      &joined_table,
      pool
  );
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  LOG(INFO) << "Join done " + std::to_string(duration.count());

  arrow::Int64Builder firstBuilder(pool);
  arrow::Int32Builder targetBuilder(pool);
  for (int i = 0; i < count; i++) {
	int l = rand() % range;
	firstBuilder.Append(i);
	targetBuilder.Append(i % 10);
  }
  std::shared_ptr<arrow::Array> firstArray;
  firstBuilder.Finish(&firstArray);
  std::shared_ptr<arrow::Int32Array> targetArray;
  targetBuilder.Finish(&targetArray);

  std::shared_ptr<std::vector<int>> ivec = std::make_shared<std::vector<int>>(10);
  int i = 0;
  std::iota(ivec->begin(), ivec->end(), 0);
  std::unique_ptr<twisterx::ArrowArraySplitKernel> kernel;
  std::shared_ptr<arrow::DataType> type = std::make_shared<arrow::Int64Type>();
  CreateSplitter(type, pool, &kernel);

  std::unordered_map<int, std::shared_ptr<arrow::Array>> out;

  LOG(INFO) << "Size: " << out.size();

  for (auto i : out) {
	LOG(INFO) << i.first << "   " << i.second->length();
	auto ids =
		std::static_pointer_cast<arrow::Int64Array>(i.second);
	for (int k = 0; k < ids->length(); k++) {
	  LOG(INFO) << "array: " << k << ": " << ids->Value(k);
	}
  }
}

std::shared_ptr<arrow::Table> make_table(int32_t id, int32_t rows, arrow::MemoryPool *memory_pool, int32_t seed) {
  Int64Builder left_id_builder(memory_pool);
  FloatBuilder cost_builder(memory_pool);

  srand(seed);

  int range = rows * 10;

  LOG(INFO) << "creating values for table " << id;

  for (int i = 0; i < rows; i++) {
	int l = rand() % range;
	float f = l + (0.1f * id);
	//LOG(INFO) << "adding " << l << " and " << f;
	left_id_builder.Append(l);
	cost_builder.Append(f);
  }

  LOG(INFO) << "created values for table " << id;

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
	  arrow::field("id", arrow::int64()), arrow::field("cost", arrow::float16())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  std::shared_ptr<arrow::Array> left_id_array;
  left_id_builder.Finish(&left_id_array);
  std::shared_ptr<arrow::Array> cost_array;
  cost_builder.Finish(&cost_array);

  return arrow::Table::Make(schema, {left_id_array, cost_array});
}

void sort_test() {
  arrow::MemoryPool *pool = arrow::default_memory_pool();

  int count = 10;

  std::shared_ptr<arrow::Table> left_table = make_table(1, 10, pool, 0);

  LOG(INFO) << "sorting...";
  std::shared_ptr<arrow::Table> sorted_table;
  twisterx::util::sort_table(left_table, 0, &sorted_table, pool);

  for (int i = 0; i < count; i++) {
	auto key = std::static_pointer_cast<arrow::Int64Array>(sorted_table->column(0)->chunk(0));
	auto val = std::static_pointer_cast<arrow::FloatArray>(sorted_table->column(1)->chunk(0));
	LOG(INFO) << "reading " << key->Value(i) << " and " << val->Value(i);
  }
}

void join_test(bool sort, int count) {
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  LOG(INFO) << "Here";
  std::shared_ptr<arrow::Table> left_table = make_table(1, count, pool, 34);
  std::shared_ptr<arrow::Table> right_table = make_table(2, count, pool, 5678);

  auto t1 = std::chrono::high_resolution_clock::now();
  arrow::Status status;
  if (sort) {
	status = twisterx::util::sort_table(left_table, 0, &left_table, pool);
	if (status != arrow::Status::OK()) {
	  LOG(FATAL) << "Failed to sort left table. " << status.ToString();
	}

	status = twisterx::util::sort_table(right_table, 0, &right_table, pool);
	if (status != arrow::Status::OK()) {
	  LOG(FATAL) << "Failed to sort right table. " << status.ToString();
	}
  }
  auto t2 = std::chrono::high_resolution_clock::now();

  LOG(INFO) << "Table sorting took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
			<< "milis";

  LOG(INFO) << "joining...";
  t1 = std::chrono::high_resolution_clock::now();
  std::shared_ptr<arrow::Table> joined_table;
  status = twisterx::join::joinTables(
      left_table,
      right_table,
      0,
      0,
      twisterx::join::JoinType::INNER,
      twisterx::join::JoinAlgorithm::SORT,
      &joined_table,
      pool
  );
  t2 = std::chrono::high_resolution_clock::now();
  count = joined_table->column(0)->length();

  if (status == arrow::Status::OK()) {
	LOG(INFO) << "Join produced " << count << " tuples  in "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "milis";
  } else {
	LOG(INFO) << "Join failed. " << status.ToString();
  }

//  for (int i = 0; i < count; i++) {
//	auto key1 = std::static_pointer_cast<arrow::Int64Array>(joined_table->column(0)->chunk(0));
//	auto val1 = std::static_pointer_cast<arrow::FloatArray>(joined_table->column(1)->chunk(0));
//	auto key2 = std::static_pointer_cast<arrow::Int64Array>(joined_table->column(2)->chunk(0));
//	auto val2 = std::static_pointer_cast<arrow::FloatArray>(joined_table->column(3)->chunk(0));
//	LOG(INFO) << "reading " << key1->Value(i) << ", " << key2->Value(i) << ", " << val1->Value(i) << ", "
//			  << val2->Value(i);
//  }
}

int main(int argc, char *argv[]) {
  join_test(true, std::atoi(argv[1]));
  join_test(false, std::atoi(argv[1]));
  return 0;
}
