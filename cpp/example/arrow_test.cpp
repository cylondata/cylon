#include <mpi.h>
#include <iostream>

#include <arrow/api.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array.h>
#include <glog/logging.h>
#include <arrow/compute/api.h>
#include "join/tx_join.cpp"

#include "arrow/arrow_all_to_all.hpp"

using arrow::DoubleBuilder;
using arrow::Int64Builder;

int main(int argc, char *argv[]) {

  arrow::MemoryPool *pool = arrow::default_memory_pool();

  Int64Builder id_builder(pool);
  Int64Builder cost_builder(pool);

  srand(10);

  for (int i = 0; i < 10; i++) {
	int r = rand() % 1000 + 1;
	//LOG(INFO) << "adding " << r;
	id_builder.Append(r);
	cost_builder.Append(i);
  }

  LOG(INFO) << "added";

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
	  arrow::field("id", arrow::int64()), arrow::field("cost", arrow::int64())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  std::shared_ptr<arrow::Array> id_array;
  id_builder.Finish(&id_array);
  std::shared_ptr<arrow::Array> cost_array;
  cost_builder.Finish(&cost_array);

  std::shared_ptr<arrow::Table> table = arrow::Table::Make(schema, {id_array, cost_array});

  twisterx::join::join<arrow::Int64Array, arrow::Int64Type, int64_t>(
	  table,
	  table,
	  0,
	  0,
	  NULLPTR,
	  NULLPTR, twisterx::join::JoinType::INNER, twisterx::join::JoinAlgorithm::SORT
  );
  return 0;
}
