#include <mpi.h>
#include <iostream>

#include <arrow/api.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array.h>
#include <glog/logging.h>
#include <arrow/compute/api.h>
#include <chrono>
#include <ctime>
#include <util/arrow_utils.h>
#include "join/tx_join.cpp"

#include "arrow/arrow_kernels.hpp"

using arrow::DoubleBuilder;
using arrow::Int64Builder;
using arrow::FloatBuilder;

void join_test() {
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

  LOG(INFO) << "Starting join";
  auto start = std::chrono::high_resolution_clock::now();
  twisterx::join::join<arrow::Int64Array, arrow::Int64Type, int64_t>(
	  left_table,
	  right_table,
	  0,
	  0,
	  NULLPTR,
	  NULLPTR, twisterx::join::JoinType::INNER, twisterx::join::JoinAlgorithm::SORT
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
  std::unique_ptr<twisterx::ArrowArrayMergeKernel> kernel;
  std::shared_ptr<arrow::DataType> type = std::make_shared<arrow::Int64Type>();
  CreateNumericMerge(type, pool, ivec, &kernel);

  std::unordered_map<int, std::shared_ptr<arrow::Array>> out;
  kernel->Merge(firstArray, targetArray, out);

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

void util_test() {
  arrow::MemoryPool *pool = arrow::default_memory_pool();

  Int64Builder left_id_builder(pool);
  FloatBuilder cost_builder(pool);

  srand(std::time(NULL));

  int count = 10;
  int range = count * 10;

  for (int i = 0; i < count; i++) {
	int l = rand() % range;
	float f = l + 0.0001f;
	LOG(INFO) << "adding " << l << " and " << f;
	left_id_builder.Append(l);
	cost_builder.Append(f);
  }

  LOG(INFO) << "added";

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
	  arrow::field("id", arrow::int64()), arrow::field("cost", arrow::float16())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  std::shared_ptr<arrow::Array> left_id_array;
  left_id_builder.Finish(&left_id_array);
  std::shared_ptr<arrow::Array> cost_array;
  cost_builder.Finish(&cost_array);

  std::shared_ptr<arrow::Table> left_table = arrow::Table::Make(schema, {left_id_array, cost_array});

  LOG(INFO) << "sorting...";
  std::shared_ptr<arrow::Table> sorted_table = twisterx::util::sort_table(left_table, 0, pool);

  for (int i = 0; i < count; i++) {
	auto key = std::static_pointer_cast<arrow::Int64Array>(sorted_table->column(0)->chunk(0));
	auto val = std::static_pointer_cast<arrow::FloatArray>(sorted_table->column(1)->chunk(0));
	LOG(INFO) << "reading " << key->Value(i) << " and " << val->Value(i);
  }

}

int main(int argc, char *argv[]) {
  util_test();
  return 0;
}
