#include <mpi.h>
#include <iostream>

#include <arrow/api.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array.h>
#include <glog/logging.h>
#include <arrow/compute/api.h>
#include <chrono>
#include <ctime>
#include "join/tx_join.cpp"

using arrow::DoubleBuilder;
using arrow::Int64Builder;

int main(int argc, char *argv[]) {
//google::InitGoogleLogging(argv[0]);

  arrow::MemoryPool *pool = arrow::default_memory_pool();

  Int64Builder left_id_builder(pool);
  Int64Builder right_id_builder(pool);
  Int64Builder cost_builder(pool);

  srand(std::time(NULL));

  int count = 100;
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

  std::vector<std::shared_ptr<arrow::Table>> tabs;
  tabs.push_back(left_table);
  tabs.push_back(right_table);

  twisterx::join::join<arrow::Int64Array, arrow::Int64Type, int64_t>(
	  tabs,
	  tabs,
	  0,
	  0,
	  NULLPTR,
	  NULLPTR, twisterx::join::JoinType::INNER, twisterx::join::JoinAlgorithm::SORT,
	  pool
  );

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
  return 0;
}
