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
#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <util/builtins.hpp>
#include <table.hpp>
#include <chrono>
#include <arrow/api.h>
#include <arrow/array.h>
#include <random>
#include <mpi.h>

#include <arrow/compute/api.h>
#include <ctx/arrow_memory_pool_utils.hpp>

#include <util/arrow_utils.hpp>
#include <groupby/groupby.hpp>

using namespace std;
using namespace arrow;

void create_table(char *const *argv,
                  arrow::MemoryPool *pool,
                  shared_ptr<arrow::Table> &left_table);

/*void aggregate(const std::function<int64_t(const int64_t &, const int64_t &)> &fun,
               const shared_ptr<arrow::Int64Array> &arr, int64_t &out) {

  const long *vals = arr->raw_values();
  const int64_t len = arr->length();

  for (int64_t i = 0; i < len; i++) {
    const int64_t &v = vals[i];
    out = fun(v, out);
  }
}

inline void aggregate2(const std::function<void(const double &, double *)> &fun,
                       const shared_ptr<arrow::DoubleArray> &arr, double *out) {

  const double *vals = arr->raw_values();
  const int64_t len = arr->length();

  for (int64_t i = 0; i < len; i++) {
    const double &v = vals[i];
    fun(v, out);
  }
}

void aggregate3(const shared_ptr<arrow::DoubleArray> &arr, double *out) {

  const double *vals = arr->raw_values();
  const int64_t len = arr->length();
  double temp = *out;
  for (int64_t i = 0; i < len; i++) {
    const double &v = vals[i];
    temp += v;
  }
  *out = temp;
}*/

void HashArrowGroupBy(arrow::MemoryPool *pool, const std::shared_ptr<cylon::Table> &ctable,
                      std::shared_ptr<cylon::Table> &output) {
  auto t1 = std::chrono::steady_clock::now();

  const shared_ptr<arrow::Table> &table = ctable->get_table();

  const shared_ptr<ChunkedArray> &idx_col = table->column(0);
  const shared_ptr<arrow::Int64Array> &index_arr = static_pointer_cast<arrow::Int64Array>(idx_col->chunk(0));
  const shared_ptr<DoubleArray> &val_col = static_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));

  arrow::Status s;

  const int64_t len = table->num_rows();

  std::unordered_map<int64_t, std::shared_ptr<arrow::DoubleBuilder>> map;
  map.reserve(len * 0.5);

  int64_t idx, count = 0;
  double val;
  for (int64_t i = 0; i < len; i++) {
    idx = index_arr->Value(i);
    val = val_col->Value(i);

    auto iter = map.find(idx);
    if (iter == map.end()) {
      auto pos = map.emplace(std::make_pair(idx, std::make_shared<arrow::DoubleBuilder>()));
      s = pos.first->second->Append(val);
    } else {
      s = iter->second->Append(val);
    }

    if (i % 100000 == 0) {
      cout << "&& " << i << endl;
    }
  }

  auto t2 = std::chrono::steady_clock::now();
  cout << "hash done! " << endl;

  shared_ptr<arrow::Array> out_idx, out_val, temp;

  arrow::Int64Builder idx_builder(pool);
  arrow::DoubleBuilder val_builder(pool);

  const unsigned long groups = map.size();
  s = idx_builder.Reserve(groups);
  s = val_builder.Reserve(groups);

  compute::FunctionContext fn_ctx;
  compute::Datum res;

  for (auto &p:  map) {
    idx_builder.UnsafeAppend(p.first);

    s = p.second->Finish(&temp);
    s = arrow::compute::Sum(&fn_ctx, temp, &res);
    p.second->Reset();
    temp.reset();

    val_builder.UnsafeAppend(static_pointer_cast<DoubleScalar>(res.scalar())->value);
  }

  temp.reset();
  map.clear();

  s = idx_builder.Finish(&out_idx);
  s = val_builder.Finish(&out_val);

  shared_ptr<Table> a_output = Table::Make(table->schema(), {out_idx, out_val});
  cylon::Table::FromArrowTable(ctable->GetContext(), a_output, &output);

  auto t3 = std::chrono::steady_clock::now();
  cout << "hash_arrow " << output->Rows()
       << " " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
       << " " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
       << endl;
}

void HashGroupBy2(const std::shared_ptr<cylon::Table> &ctable,
                  std::shared_ptr<cylon::Table> &output,
                  const std::function<void(const double &, double *)> &fun) {
  auto t1 = std::chrono::steady_clock::now();

  const shared_ptr<arrow::Table> &table = ctable->get_table();

  const shared_ptr<ChunkedArray> &idx_col = table->column(0);
  const shared_ptr<arrow::Int64Array> &index_arr = static_pointer_cast<arrow::Int64Array>(idx_col->chunk(0));
  const shared_ptr<DoubleArray> &val_col = static_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));

  arrow::Status s;

  const int64_t len = table->num_rows();

  std::unordered_map<int64_t, std::tuple<double>> map;
  map.reserve(len * 0.5);

  int64_t idx;
  double val;
  for (int64_t i = 0; i < len; i++) {
    idx = index_arr->Value(i);
    val = val_col->Value(i);

    auto iter = map.find(idx);

    if (iter == map.end()) {
//      map.insert(make_pair(idx, val));
      auto pos = map.emplace(idx, 0);
      fun(val, &(std::get<0>(pos.first->second)));
    } else {
      fun(val, &(std::get<0>(iter->second))); // update the value using the fun
    }
  }

  auto t2 = std::chrono::steady_clock::now();

  arrow::Int64Builder idx_builder;
  arrow::DoubleBuilder val_builder;
  shared_ptr<arrow::Array> out_idx, out_val;

  const unsigned long groups = map.size();
  s = idx_builder.Reserve(groups);
  s = val_builder.Reserve(groups);

  for (auto &p:  map) {
    idx_builder.UnsafeAppend(p.first);
    val_builder.UnsafeAppend(std::get<0>(p.second));
  }
  map.clear();

  s = idx_builder.Finish(&out_idx);
  s = val_builder.Finish(&out_val);

  shared_ptr<Table> a_output = Table::Make(table->schema(), {out_idx, out_val});
  cylon::Table::FromArrowTable(ctable->GetContext(), a_output, &output);

  auto t3 = std::chrono::steady_clock::now();
  cout << "hash_group " << output->Rows()
       << " " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
       << " " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
       << endl;
}

void HashGroupBy(const std::shared_ptr<cylon::Table> &ctable,
                 std::shared_ptr<cylon::Table> &output,
                 const std::function<void(const double &, double *)> &fun) {
  auto t1 = std::chrono::steady_clock::now();

  const shared_ptr<arrow::Table> &table = ctable->get_table();

  const shared_ptr<ChunkedArray> &idx_col = table->column(0);
  const shared_ptr<arrow::Int64Array> &index_arr = static_pointer_cast<arrow::Int64Array>(idx_col->chunk(0));
  const shared_ptr<DoubleArray> &val_col = static_pointer_cast<arrow::DoubleArray>(table->column(1)->chunk(0));

  arrow::Status s;

  const int64_t len = table->num_rows();

  std::unordered_map<int64_t, double> map;
  map.reserve(len * 0.5);

  int64_t idx;
  double val;
  for (int64_t i = 0; i < len; i++) {
    idx = index_arr->Value(i);
    val = val_col->Value(i);

    auto iter = map.find(idx);

    if (iter == map.end()) {
      map.insert(make_pair(idx, val));
    } else {
      fun(val, &(iter->second)); // update the value using the fun
    }
  }

  auto t2 = std::chrono::steady_clock::now();

  arrow::Int64Builder idx_builder;
  arrow::DoubleBuilder val_builder;
  shared_ptr<arrow::Array> out_idx, out_val;

  const unsigned long groups = map.size();
  s = idx_builder.Reserve(groups);
  s = val_builder.Reserve(groups);

  for (auto &p:  map){
    idx_builder.UnsafeAppend(p.first);
    val_builder.UnsafeAppend(p.second);
  }
  map.clear();

  s = idx_builder.Finish(&out_idx);
  s = val_builder.Finish(&out_val);

  shared_ptr<Table> a_output = Table::Make(table->schema(), {out_idx, out_val});
  cylon::Table::FromArrowTable(ctable->GetContext(), a_output, &output);

  auto t3 = std::chrono::steady_clock::now();
  cout << "hash_group " << output->Rows()
       << " " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
       << " " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
       << endl;
}

void HashGroupBy3(arrow::MemoryPool *pool, const std::shared_ptr<cylon::Table> &ctable,
                  std::shared_ptr<cylon::Table> &output) {
  auto t1 = std::chrono::steady_clock::now();

  const shared_ptr<arrow::Table> &table = ctable->get_table();

  std::vector<shared_ptr<arrow::Array>> cols;

  cylon::Status s = cylon::HashGroupBy<arrow::Int64Type, arrow::DoubleType, cylon::GroupByAggregationOp::MEAN>
      (pool, table->column(0), table->column(1), cols);

  shared_ptr<Table> a_output = Table::Make(table->schema(), cols);
  cylon::Table::FromArrowTable(ctable->GetContext(), a_output, &output);

  auto t3 = std::chrono::steady_clock::now();
  cout << "hash_group3 " << output->Rows()
       << " " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1).count()
       //       << " " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
       << endl;
}

void ArrowGroupBy(const std::shared_ptr<cylon::Table> &ctable, std::shared_ptr<cylon::Table> &output) {
  shared_ptr<cylon::Table> sorted_table;
  auto t1 = std::chrono::steady_clock::now();
  ctable->Sort(0, sorted_table);
  auto t2 = std::chrono::steady_clock::now();

  const shared_ptr<arrow::Table> &table = sorted_table->get_table();

  const shared_ptr<ChunkedArray> &idx_col = table->column(0);
  const shared_ptr<arrow::Int64Array> &index_arr = static_pointer_cast<arrow::Int64Array>(idx_col->chunk(0));
  arrow::Int64Builder idx_builder;
  shared_ptr<arrow::Array> out_idx;
  arrow::Status s;

  vector<int64_t> boundaries;
  const int64_t len = table->num_rows();
  boundaries.reserve((int64_t) len * 0.99);
  int64_t prev_v = index_arr->Value(0), curr_v;
  for (int64_t i = 0; i < len; i++) {
    curr_v = index_arr->Value(i);

    if (curr_v != prev_v) {
      boundaries.push_back(i);
      s = idx_builder.Append(prev_v);
      prev_v = curr_v;
    }
  }
  boundaries.push_back(len);
  s = idx_builder.Append(prev_v);

  s = idx_builder.Finish(&out_idx);

  const shared_ptr<Array> &val_col = table->column(1)->chunk(0);
  arrow::DoubleBuilder val_builder;
  s = val_builder.Reserve(boundaries.size());
  shared_ptr<arrow::Array> out_val;

  compute::FunctionContext fn_ctx;
  compute::Datum res;
  int64_t start = 0;
  for (auto &end: boundaries) {
    s = compute::Sum(&fn_ctx, val_col->Slice(start, end - start), &res);
    start = end;

    val_builder.UnsafeAppend(static_pointer_cast<DoubleScalar>(res.scalar())->value);
  }
  s = val_builder.Finish(&out_val);

  shared_ptr<Table> a_output = Table::Make(table->schema(), {out_idx, out_val});

  sorted_table.reset(); // release the sorted table

  cylon::Table::FromArrowTable(ctable->GetContext(), a_output, &output);

  auto t3 = std::chrono::steady_clock::now();

  cout << "arrow_group " << output->Rows()
       << " " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
       << " " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
       << endl;
//  sorted_table->Print();
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    LOG(ERROR) << "There should be 2 args. count, duplication factor";
    return 1;
  }

  auto start_start = std::chrono::steady_clock::now();
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::Table> left_table;
//  create_binary_table(argv, ctx, pool, left_table, right_table);
  create_table(argv, pool, left_table);
  MPI_Barrier(MPI_COMM_WORLD);

  std::shared_ptr<cylon::Table> first_table;
  auto status = cylon::Table::FromArrowTable(ctx, left_table, &first_table);
  if (!status.is_ok()) {
    LOG(INFO) << "Table reading failed " << argv[1];
    ctx->Finalize();
    return 1;
  }

  auto read_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Read tables in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                read_end_time - start_start).count() << "[ms]";

//  first_table->Sort(0, sorted_table);
  first_table->Print();

//  const shared_ptr<arrow::Table> &a_tab = sorted_table->get_table();
  shared_ptr<cylon::Table> output;

//  ArrowGroupBy(first_table, output);
//  output->Print();

  output.reset();
  cout << "++++++++++++++++++++++++++" << endl;

  auto sum = [](const double &v, double *out) -> void {
    *out += v;
  };

//  HashGroupBy(first_table, output, sum);
////  output->Print();
//  output.reset();

  cout << "++++++++++++++++++++++++++" << endl;

  HashGroupBy3(pool, first_table, output);
  output->Print();
  output.reset();

  cout << "++++++++++++++++++++++++++" << endl;

//  HashArrowGroupBy(pool, first_table, output);
//  output->Print();
//  output.reset();
////  output1->Print();


  /* const shared_ptr<arrow::ChunkedArray> &col = first_table->get_table()->column(1);

   arrow::compute::FunctionContext fn_ctx(cylon::ToArrowPool(ctx));
   const arrow::compute::Datum input = arrow::compute::Datum(col);
   arrow::compute::Datum output;

   auto join_end_time = std::chrono::steady_clock::now();
   arrow::Status status1 = arrow::compute::Sum(&fn_ctx, input, &output);
   auto join_end_time1 = std::chrono::steady_clock::now();

   const shared_ptr<arrow::DoubleScalar> &oo = std::static_pointer_cast<arrow::DoubleScalar>(output.scalar());

   LOG(INFO) << "sum " << oo->value << " "
             << std::chrono::duration_cast<std::chrono::milliseconds>(
                 join_end_time1 - join_end_time).count();

   auto sum = [](const double &v, double *out) -> void {
     *out += v;
   };


 //  const std::function<int64_t(const int64_t &, const int64_t &)> sum = [](const int64_t &v, const int64_t &out) ->
 //      int64_t {
 //    return out + v;
 //  };

   double out = 0;

   join_end_time = std::chrono::steady_clock::now();
   aggregate2(sum, std::static_pointer_cast<arrow::DoubleArray>(col->chunk(0)), &out);
 //  aggregate(sum, std::static_pointer_cast<arrow::Int64Array>(col->chunk(0)), out);
 //  aggregate3(std::static_pointer_cast<arrow::Int64Array>(col->chunk(0)), &out);
   join_end_time1 = std::chrono::steady_clock::now();

   LOG(INFO) << "sum2 " << out << " "
             << std::chrono::duration_cast<std::chrono::milliseconds>(
                 join_end_time1 - join_end_time).count();

   out = 0;
   join_end_time = std::chrono::steady_clock::now();
   aggregate3(std::static_pointer_cast<arrow::DoubleArray>(col->chunk(0)), &out);
   join_end_time1 = std::chrono::steady_clock::now();

   LOG(INFO) << "sum3 " << out << " "
             << std::chrono::duration_cast<std::chrono::milliseconds>(
                 join_end_time1 - join_end_time).count();
 */
  ctx->Finalize();
  return 0;
}

void create_table(char *const *argv,
                  arrow::MemoryPool *pool,
                  shared_ptr<arrow::Table> &left_table) {
  arrow::Int64Builder left_id_builder(pool);
  arrow::DoubleBuilder cost_builder(pool);
  uint64_t count = stoull(argv[1]);
  double dup = stod(argv[2]);

  cout << "#### lines " << count << " dup " << dup << endl;

  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<int64_t> distrib(0, (int64_t) (count * dup));

  std::mt19937_64 gen1(rd());
  std::uniform_real_distribution<double> distrib1;

//  uint64_t range = count * ctx->GetWorldSize();
//  srand(time(NULL) + ctx->GetRank());

  arrow::Status st = left_id_builder.Reserve(count);
  st = cost_builder.Reserve(count);
  for (uint64_t i = 0; i < count; i++) {
    int64_t l = distrib(gen);
    double v = distrib1(gen1);
    left_id_builder.UnsafeAppend(l);
    cost_builder.UnsafeAppend(v);
  }

  shared_ptr<arrow::Array> left_id_array;
  shared_ptr<arrow::Array> cost_array;

  st = left_id_builder.Finish(&left_id_array);
  st = cost_builder.Finish(&cost_array);

  vector<shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("first", arrow::int64()),
      arrow::field("second", arrow::float64())};
  auto schema = make_shared<arrow::Schema>(schema_vector);

  left_table = arrow::Table::Make(schema, {std::move(left_id_array), std::move(cost_array)});

}
