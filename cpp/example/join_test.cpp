#include <mpi.h>
#include <iostream>

#include <arrow/array/builder_primitive.h>
#include <arrow/array.h>
#include <glog/logging.h>
#include <arrow/compute/api.h>
#include <chrono>
#include <ctime>
#include "arrow/arrow_join.hpp"

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
  bool onJoin(std::shared_ptr <arrow::Table> table) override {
    LOG(INFO) << "Joined";
    return true;
  }
};

int main(int argc, char *argv[]) {
  MPI_Init(NULLPTR, NULLPTR);
  arrow::MemoryPool *pool =arrow::default_memory_pool();

//  int* x = (int *)malloc(10 * sizeof(int));
//  std::cout << "error: " << x << std::endl;
//  x[12] = 1;

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Int64Builder left_id_builder(pool);
  Int64Builder right_id_builder(pool);
  Int64Builder cost_builder(pool);

  srand(std::time(NULL) + rank);

  int count = std::atoi(argv[1]) / size;
  LOG(INFO) << "No of tuples " << count;
  int range = count * size;
  std::vector<std::shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("id", arrow::int64()), arrow::field("cost", arrow::int64())};
  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  std::vector<int> sources;
  std::vector<int> targets;
  for (int i = 0; i < size; i++) {
    sources.push_back(i);
    targets.push_back(i);
  }

  int actualCount = std::atoi(argv[1]);
  long* values = new long[actualCount];
  int* indices = new int[actualCount];
  for (int i = 0; i < actualCount; i++) {
    indices[i] = i;
    int l = rand() % range;
    values[i] = l;
  }
  auto start2 = std::chrono::high_resolution_clock::now();
  std::stable_sort(indices, indices + actualCount, [values](uint64_t left, uint64_t right) {
    return values[left] < values[right];
  });

  auto end2 = std::chrono::high_resolution_clock::now();
  auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
  LOG(INFO) << "Sort done 1 " + std::to_string(duration3.count());

  auto start3 = std::chrono::high_resolution_clock::now();
  std::stable_sort(values, values + actualCount);

  auto end3 = std::chrono::high_resolution_clock::now();
  auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
  LOG(INFO) << "Sort done 2 " + std::to_string(duration4.count());
  delete [] values;
  delete [] indices;

  JC jc;
  twisterx::ArrowJoin join(rank, sources, targets, 0, 1, &jc, schema, pool);
  auto start = std::chrono::high_resolution_clock::now();
  for (int j = 0; j < size; j++) {
    for (int i = 0; i < count; i++) {
      int l = rand() % range;
      int r = rand() % range;

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

    join.leftInsert(left_table, (j + rank) % size);
    join.rightInsert(right_table, (j + rank) % size);

    // call this to progress comms
    join.isComplete();
  }

  auto start1 = std::chrono::high_resolution_clock::now();
  join.finish();
  while (!join.isComplete()) {
  }
  join.close();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start1);
  LOG(INFO) << "Join done " + std::to_string(duration.count()) << " comm: " << std::to_string(duration2.count());

  MPI_Finalize();
  return 0;
}