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
#include "example_utils.hpp"

namespace cylon {
namespace examples {

void create_table(uint64_t count, double dup,
                        std::shared_ptr<cylon::CylonContext> &ctx,
                        arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::Table> &left_table) {
  arrow::Int64Builder left_id_builder(pool);
  arrow::DoubleBuilder cost_builder(pool);
  std::random_device rd;
  std::mt19937_64 gen(0);
  std::uniform_int_distribution<int64_t> distrib(0, (int64_t) (count * dup * ctx->GetWorldSize()));

  std::mt19937_64 gen1(rd());
  std::uniform_real_distribution<double> val_distrib;

  arrow::Status st = left_id_builder.Reserve(count);
  st = cost_builder.Reserve(count);
  for (uint64_t i = 0; i < count; i++) {
    int64_t l = distrib(gen);
    double v = val_distrib(gen1);
    left_id_builder.UnsafeAppend(l);
    cost_builder.UnsafeAppend(v);
  }

  std::shared_ptr<arrow::Array> left, vals;
  st = left_id_builder.Finish(&left);
  st = cost_builder.Finish(&vals);
  std::vector<std::shared_ptr<arrow::Field>> left_schema_vector = {arrow::field("first", left->type()),
                                                                   arrow::field("second", vals->type())};
  auto left_schema = std::make_shared<arrow::Schema>(left_schema_vector);
  left_table = arrow::Table::Make(left_schema, {std::move(left), vals});
}

void create_int64_table(uint64_t count, double dup,
                        std::shared_ptr<cylon::CylonContext> &ctx,
                        arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::Table> &left_table,
                        std::shared_ptr<arrow::Table> &right_table) {
  arrow::Int64Builder left_id_builder(pool), right_id_builder(pool);
  arrow::DoubleBuilder cost_builder(pool);
  std::random_device rd;
  std::mt19937_64 gen(0);
  std::uniform_int_distribution<int64_t> distrib(0, (int64_t) (count * dup * ctx->GetWorldSize()));

  std::mt19937_64 gen1(rd());
  std::uniform_real_distribution<double> val_distrib;

  arrow::Status st = left_id_builder.Reserve(count);
  st = right_id_builder.Reserve(count);
  st = cost_builder.Reserve(count);
  for (uint64_t i = 0; i < count; i++) {
    int64_t l = distrib(gen);
    int64_t r = distrib(gen);
    double v = val_distrib(gen1);
    left_id_builder.UnsafeAppend(l);
    right_id_builder.UnsafeAppend(r);
    cost_builder.UnsafeAppend(v);
  }

  std::shared_ptr<arrow::Array> left, right, vals;
  st = left_id_builder.Finish(&left);
  st = right_id_builder.Finish(&right);
  st = cost_builder.Finish(&vals);
  std::vector<std::shared_ptr<arrow::Field>> left_schema_vector = {arrow::field("first", left->type()),
                                                                   arrow::field("second", vals->type())};
  auto left_schema = std::make_shared<arrow::Schema>(left_schema_vector);
  left_table = arrow::Table::Make(left_schema, {std::move(left), vals});
  std::vector<std::shared_ptr<arrow::Field>> right_schema_vector = {arrow::field("first", right->type()),
                                                                    arrow::field("second", vals->type())};
  auto right_schema = std::make_shared<arrow::Schema>(right_schema_vector);
  right_table = arrow::Table::Make(right_schema, {std::move(right), std::move(vals)});
}

int create_two_in_memory_tables_from_arrow_tables(std::shared_ptr<cylon::CylonContext> &ctx,
                                                   std::shared_ptr<arrow::Table> left_table,
                                                   std::shared_ptr<arrow::Table> right_table,
                                                   std::shared_ptr<cylon::Table> &first_table,
                                                   std::shared_ptr<cylon::Table> &second_table) {
  auto status = cylon::Table::FromArrowTable(ctx, left_table, first_table);
  if (!status.is_ok()) {
    return 1;
  }

  status = cylon::Table::FromArrowTable(ctx, right_table, second_table);
  if (!status.is_ok()) {
    return 1;
  }
  return 0;
}

int create_two_in_memory_tables(uint64_t count, double dup,
                                 std::shared_ptr<cylon::CylonContext> &ctx,
                                 std::shared_ptr<cylon::Table> &first_table,
                                 std::shared_ptr<cylon::Table> &second_table) {
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::Table> left_table, right_table;
  cylon::examples::create_int64_table(count, dup, ctx, pool, left_table, right_table);
  create_two_in_memory_tables_from_arrow_tables(ctx, left_table, right_table, first_table, second_table);
  return 0;
}

int create_in_memory_tables(uint64_t count, double dup,
                                std::shared_ptr<cylon::CylonContext> &ctx,
                                std::shared_ptr<cylon::Table> &first_table) {
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::Table> left_table, right_table;
  cylon::examples::create_table(count, dup, ctx, pool, left_table);
  auto status = cylon::Table::FromArrowTable(ctx, left_table, first_table);
  if (!status.is_ok()) {
    return 1;
  }
  return 0;
}


} // end namespace examples
} // end namespace cylon
