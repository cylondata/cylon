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

#include <utility>
#include "util/arrow_rand.hpp"

namespace cylon {
namespace examples {

void create_table(int64_t count, double dup,
                  std::shared_ptr<cylon::CylonContext> &ctx,
                  arrow::MemoryPool *pool,
                  std::shared_ptr<arrow::Table> &left_table,
                  double null_prob) {
  RandomArrayGenerator gen(/*seed=*/0, pool);
  auto left = gen.Numeric<arrow::Int64Type>(count, 0LL,
                                            (int64_t) (count * ctx->GetWorldSize() * dup),
                                            null_prob);
  auto vals = gen.Numeric<arrow::DoubleType>(count, 0.0, 1.0, null_prob);

  auto left_schema = arrow::schema({arrow::field("first", left->type()),
                                    arrow::field("second", vals->type())});
  left_table = arrow::Table::Make(left_schema, {std::move(left), std::move(vals)});
}

void create_int64_table(int64_t count, double dup,
                        std::shared_ptr<cylon::CylonContext> &ctx,
                        arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::Table> &left_table,
                        std::shared_ptr<arrow::Table> &right_table,
                        double null_prob) {
  RandomArrayGenerator gen(/*seed=*/0, pool);

  auto max = (int64_t) (count * dup * ctx->GetWorldSize());
  auto left = gen.Numeric<arrow::Int64Type>(count, 0LL, max, null_prob);
  auto right = gen.Numeric<arrow::Int64Type>(count, 0LL, max, null_prob);
  auto vals = gen.Numeric<arrow::DoubleType>(count, 0.0, 1.0, null_prob);

  auto left_schema = arrow::schema({arrow::field("first", left->type()),
                                    arrow::field("second", vals->type())});
  left_table = arrow::Table::Make(left_schema, {std::move(left), vals});

  auto right_schema = arrow::schema({arrow::field("first", right->type()),
                                     arrow::field("second", vals->type())});
  right_table = arrow::Table::Make(right_schema, {std::move(right), std::move(vals)});
}

int create_two_in_memory_tables_from_arrow_tables(std::shared_ptr<cylon::CylonContext> &ctx,
                                                   std::shared_ptr<arrow::Table> left_table,
                                                   std::shared_ptr<arrow::Table> right_table,
                                                   std::shared_ptr<cylon::Table> &first_table,
                                                   std::shared_ptr<cylon::Table> &second_table) {
  auto status = cylon::Table::FromArrowTable(ctx, std::move(left_table), first_table);
  if (!status.is_ok()) {
    return 1;
  }

  status = cylon::Table::FromArrowTable(ctx, std::move(right_table), second_table);
  if (!status.is_ok()) {
    return 1;
  }
  return 0;
}

int create_two_in_memory_tables(int64_t count,
                                double dup,
                                std::shared_ptr<cylon::CylonContext> &ctx,
                                std::shared_ptr<cylon::Table> &first_table,
                                std::shared_ptr<cylon::Table> &second_table,
                                double null_prob) {
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::Table> left_table, right_table;
  create_int64_table(count, dup, ctx, pool, left_table, right_table, null_prob);
  create_two_in_memory_tables_from_arrow_tables(ctx,
                                                left_table,
                                                right_table,
                                                first_table,
                                                second_table);
  return 0;
}

int create_in_memory_tables(int64_t count,
                            double dup,
                            std::shared_ptr<cylon::CylonContext> &ctx,
                            std::shared_ptr<cylon::Table> &first_table,
                            double null_prob) {
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::Table> left_table, right_table;
  create_table(count, dup, ctx, pool, left_table, null_prob);
  auto status = cylon::Table::FromArrowTable(ctx, left_table, first_table);
  if (!status.is_ok()) {
    return 1;
  }
  return 0;
}


} // end namespace examples
} // end namespace cylon
