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

#ifndef CYLON_SRC_CYLON_ARROW_ARROW_COMPARATOR_HPP_
#define CYLON_SRC_CYLON_ARROW_ARROW_COMPARATOR_HPP_

#include <arrow/api.h>

#include "../ctx/cylon_context.hpp"
#include "arrow_partition_kernels.hpp"

namespace cylon {

class ArrowComparator {
 public:
  virtual int compare(const std::shared_ptr<arrow::Array> &array1, int64_t index1,
                      const std::shared_ptr<arrow::Array> &array2, int64_t index2) = 0;
};

std::shared_ptr<ArrowComparator> GetComparator(const std::shared_ptr<arrow::DataType> &type);

// -----------------------------------------------------------------------------

/**
 * This Util class can be used to compare the equality  of rows of two tables. This class
 * expect Tables to have just a single chunk in all its columns
 *
 * todo resolve single chunk expectation
 */
class TableRowComparator {
 private:
  std::vector<std::shared_ptr<ArrowComparator>> comparators;
 public:
  explicit TableRowComparator(const std::vector<std::shared_ptr<arrow::Field>> &vector);
  int compare(const std::shared_ptr<arrow::Table> &table1, int64_t index1,
              const std::shared_ptr<arrow::Table> &table2, int64_t index2);
};

// -----------------------------------------------------------------------------

class ArrayIndexComparator {
 public:
  virtual int compare(int64_t index1, int64_t index2) const = 0;
};

std::shared_ptr<ArrayIndexComparator> CreateArrayIndexComparator(const std::shared_ptr<arrow::Array> &array, bool asc = true);

// -----------------------------------------------------------------------------

class RowComparator {
 public:
  RowComparator(std::shared_ptr<cylon::CylonContext> &ctx,
                const std::shared_ptr<arrow::Table> *tables,
                int64_t *eq,
                int64_t *hs);

  // equality
  bool operator()(const std::pair<int8_t, int64_t> &record1, const std::pair<int8_t, int64_t> &record2) const;

  // hashing
  size_t operator()(const std::pair<int8_t, int64_t> &record) const;

 private:
  const std::shared_ptr<arrow::Table> *tables;
  std::shared_ptr<cylon::TableRowComparator> comparator;
  std::shared_ptr<cylon::RowHashingKernel> row_hashing_kernel;
  int64_t *eq, *hs;
};

// -----------------------------------------------------------------------------

/**
 * comparator to compare indices within a table based on multiple column indices
 */
class TableRowIndexComparator {
 public:
  TableRowIndexComparator(const std::shared_ptr<arrow::Table> &table, const std::vector<int> &col_ids);

  // equality
  bool operator()(const int64_t &record1, const int64_t &record2) const;

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  // hence they are wrapped around smart pointers
  std::shared_ptr<std::vector<std::shared_ptr<ArrayIndexComparator>>> idx_comparators_ptr;
};

// -----------------------------------------------------------------------------

/**
 * hash index within a table based on multiple column indices.
 * Note: A composite hash would be precomputed in the constructor
 */
class TableRowIndexHash {
 public:
  TableRowIndexHash(const std::shared_ptr<arrow::Table> &table, const std::vector<int> &col_ids);

  // hashing
  size_t operator()(const int64_t &record) const;

  /**
   * Get the composite hashes as arrow::Array
   * @param hasher
   * @return
   */
  static std::shared_ptr<arrow::UInt32Array> GetHashArray(const TableRowIndexHash &hasher);

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  // hence they are wrapped around smart pointers
  std::shared_ptr<std::vector<uint32_t>> hashes_ptr;
};

// -----------------------------------------------------------------------------

/**
 * multi-table hashes based on a <table index, row index> pair
 */
class MultiTableRowIndexHash {
 public:
  explicit MultiTableRowIndexHash(const std::vector<std::shared_ptr<arrow::Table>> &tables);

  // hashing
  size_t operator()(const std::pair<int8_t, int64_t> &record) const;

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  std::shared_ptr<std::vector<TableRowIndexHash>> hashes_ptr;
};

// -----------------------------------------------------------------------------

/**
 * multi-table comparator based on a <table index, row index> pair
 */
class MultiTableRowIndexComparator {
 public:
  explicit MultiTableRowIndexComparator(const std::vector<std::shared_ptr<arrow::Table>> &tables);

  bool operator()(const std::pair<int8_t, int64_t> &record1, const std::pair<int8_t, int64_t> &record2) const;

 private:
  const std::vector<std::shared_ptr<arrow::Table>> &tables;
  std::shared_ptr<TableRowComparator> comparator;
};

}  // namespace cylon

#endif //CYLON_SRC_CYLON_ARROW_ARROW_COMPARATOR_HPP_
