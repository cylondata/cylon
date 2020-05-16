#include <mpi.h>
#include <iostream>

#include <arrow/array/builder_primitive.h>
#include <arrow/array.h>
#include <glog/logging.h>
#include <arrow/compute/api.h>
#include <chrono>
#include <ctime>
#include <net/mpi/mpi_communicator.h>
#include "arrow/arrow_join.hpp"
#include "table.hpp"

using arrow::DoubleBuilder;
using arrow::Int64Builder;

class JC : public twisterx::JoinCallback {
 public:
  /**
  * This function is called when a data is received
  * @param source the source
  * @param buffer the buffer allocated by the system, we need to free this
  * @param length the length of the buffer
  * @return true if we accept this buffer
  */
  bool onJoin(std::shared_ptr<arrow::Table> table) override {
	LOG(INFO) << "Joined";
	return true;
  }
};

int main(int argc, char *argv[]) {

  //auto mpi_config = new twisterx::net::MPIConfig();
  //auto ctx = twisterx::TwisterXContext::InitDistributed(mpi_config);

  int rank = 0;//ctx->GetRank();
  int size = 1;//ctx->GetWorldSize();

  std::vector<int> sources;
  std::vector<int> targets;
  for (int i = 0; i < size; i++) {
	sources.push_back(i);
	targets.push_back(i);
  }
  int actualCount = std::atoi(argv[1]);
  int count = std::atoi(argv[1]) / size;
  if (rank == 0) {
	LOG(INFO) << "No of tuples per worker :" << count << ", Total Tuples : " << actualCount;
  }

  int range = count * size;

  arrow::MemoryPool *pool = arrow::default_memory_pool();

  Int64Builder left_id_builder(pool);
  Int64Builder right_id_builder(pool);
  Int64Builder cost_builder(pool);

  srand(std::time(NULL));

  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
	  arrow::field("id", arrow::int64()),
	  arrow::field("cost", arrow::int64())
  };

  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  long *values = new long[actualCount];
  int *indices = new int[actualCount];
  for (int i = 0; i < actualCount; i++) {
	indices[i] = i;
	int l = rand() % range;
	values[i] = l;
  }
//  auto start2 = std::chrono::high_resolution_clock::now();
//  std::stable_sort(indices, indices + actualCount, [values](uint64_t left, uint64_t right) {
//	return values[left] < values[right];
//  });
//
//  auto end2 = std::chrono::high_resolution_clock::now();
//  auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
//
//  if (rank == 0) {
//	LOG(INFO) << "Sort done 1 " + std::to_string(duration3.count());
//  }
//
//  auto start3 = std::chrono::high_resolution_clock::now();
//  std::stable_sort(values, values + actualCount);
//
//  auto end3 = std::chrono::high_resolution_clock::now();
//  auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
//  if (rank == 0) {
//	LOG(INFO) << "Sort done 2 " + std::to_string(duration4.count());
//  }
  delete[] values;
  delete[] indices;

  auto start = std::chrono::high_resolution_clock::now();
  long genTime = 0;

  auto genTimeStart = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < count; i++) {
	int l = rand();
	int r = rand();

	left_id_builder.Append(l);
	right_id_builder.Append(r);
	cost_builder.Append(i);
  }

  std::shared_ptr<arrow::Array> left_id_array;
  left_id_builder.Finish(&left_id_array);
  std::shared_ptr<arrow::Array> right_id_array;
  right_id_builder.Finish(&right_id_array);

  std::shared_ptr<arrow::Array> cost_array;
  cost_builder.Finish(&cost_array);

  std::shared_ptr<arrow::Table> left_table = arrow::Table::Make(schema, {left_id_array, cost_array});
  std::shared_ptr<arrow::Table> right_table = arrow::Table::Make(schema, {right_id_array, cost_array});
  auto genTimeEnd = std::chrono::high_resolution_clock::now();

  std::shared_ptr<twisterx::Table> left_table_tx;
  std::shared_ptr<twisterx::Table> right_table_tx;
  std::shared_ptr<twisterx::Table> joined_tx;

  std::cout << "Rows : " << left_table->num_rows() << ", Columns : " << left_table->num_columns() << std::endl;

  auto status_lf_tx = twisterx::Table::FromArrowTable(left_table, &left_table_tx);
  auto status_rt_tx = twisterx::Table::FromArrowTable(left_table, &right_table_tx);

  std::cout << "Left arrow table Converted to tx table  : " << status_lf_tx.get_msg() << std::endl;
  std::cout << "Right arrow table Converted to tx table  : " << status_rt_tx.get_msg() << std::endl;

  std::cout << "Left Table Id : " << left_table_tx->get_id() << std::endl;
  std::cout << "Right Table Id : " << right_table_tx->get_id() << std::endl;

//	std::cout << "Left Table" << std::endl;
//	left_table_tx->print();
//	std::cout << "Left Table" << std::endl;
//	right_table_tx->print();
  auto start_join = std::chrono::high_resolution_clock::now();
  auto reps = 10;
  for (int i = 0; i < reps; i++) {
	left_table_tx->Join(right_table_tx,
						twisterx::join::config::JoinConfig::InnerJoin(0, 0),
						&joined_tx);
  }
  auto end_join = std::chrono::high_resolution_clock::now();
  auto join_duration = std::chrono::duration_cast<std::chrono::milliseconds>((end_join - start_join) / reps);
  std::cout << "[Joined Table] : Rows : " << joined_tx->rows() << ", Columns : " << joined_tx->columns() << std::endl;

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  if (rank == 0) {
	LOG(INFO) << "Join time : " + std::to_string(join_duration.count()) << " ms";
	LOG(INFO) << "Total time " + std::to_string(duration.count()) << " genTime : " << std::to_string(genTime);
  }

  return 0;
}