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

#ifndef CYLON_CPP_SRC_CYLON_COMPUTE_AGGREGATES_HPP_
#define CYLON_CPP_SRC_CYLON_COMPUTE_AGGREGATES_HPP_

#include <arrow/compute/api.h>

#include <utility>
#include <status.hpp>
#include <table.hpp>
#include <ctx/arrow_memory_pool_utils.hpp>

namespace cylon {
namespace compute {

/**
 * Container to hold aggregation results
 */
class Result {
 public:
  /**
   * move the ownership of the arrow datum result to Result container
   * @param result
   */
  explicit Result(arrow::compute::Datum result) : result(std::move(result)) {

  }

  /**
   * Return a const reference to the underlying datum object
   * @return
   */
  const arrow::compute::Datum &GetResult() const {
    return result;
  }

 private:
  const arrow::compute::Datum result;
};

/**
 * Function pointer for aggregate functions
 */
typedef Status
(*AggregateOperation)(const std::shared_ptr<cylon::Table> &table, int32_t col_idx, std::shared_ptr<Result> &output);

/**
 * Calculates the global sum of a column
 * @param ctx
 * @param table
 * @param col_idx
 * @param output
 * @return
 */
cylon::Status Sum(const std::shared_ptr<cylon::Table> &table, int32_t col_idx, std::shared_ptr<Result> &output);

/**
 * Calculates the global count of a column
 * @param ctx
 * @param table
 * @param col_idx
 * @param output
 * @return
 */
cylon::Status Count(const std::shared_ptr<cylon::Table> &table, int32_t col_idx, std::shared_ptr<Result> &output);

/**
 * Calculates the global min of a column
 * @param ctx
 * @param table
 * @param col_idx
 * @param output
 * @return
 */
cylon::Status Min(const std::shared_ptr<cylon::Table> &table, int32_t col_idx, std::shared_ptr<Result> &output);

/**
 * Calculates the global max of a column
 * @param ctx
 * @param table
 * @param col_idx
 * @param output
 * @return
 */
cylon::Status Max(const std::shared_ptr<cylon::Table> &table, int32_t col_idx, std::shared_ptr<Result> &output);

cylon::Status Sum(const std::shared_ptr<cylon::Table> &table,
                  int32_t col_idx,
                  std::shared_ptr<cylon::Table> &output);

cylon::Status Count(const std::shared_ptr<cylon::Table> &table,
                    int32_t col_idx,
                    std::shared_ptr<cylon::Table> &output);

cylon::Status Min(const std::shared_ptr<cylon::Table> &table,
                  int32_t col_idx,
                  std::shared_ptr<cylon::Table> &output);

cylon::Status Max(const std::shared_ptr<cylon::Table> &table,
                  int32_t col_idx,
                  std::shared_ptr<cylon::Table> &output);

} // end compute
} // end cylon

#endif //CYLON_CPP_SRC_CYLON_COMPUTE_AGGREGATES_HPP_
