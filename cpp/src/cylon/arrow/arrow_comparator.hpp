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

/**
 * @deprecated
 */
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
 *
 * @deprecated
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

/**
 * To compare indices in a single arrays
 */
class ArrayIndexComparator {
 public:
  virtual int compare(int64_t index1, int64_t index2) const = 0;
  virtual bool equal_to(int64_t index1, int64_t index2) const = 0;
};

/**
 * Creates a comparator for a single array
 * @param array
 * @param asc
 * @return
 */
std::shared_ptr<ArrayIndexComparator> CreateArrayIndexComparator(const std::shared_ptr<arrow::Array> &array,
                                                                 bool asc = true);

// -----------------------------------------------------------------------------

/**
 * To compare indices in two arrays
 */
class TwoArrayIndexComparator {
 public:
  /**
   * compare two indices.
   * IMPORTANT: to uniquely identify arrays, the most significant bit of index value is encoded as,
   *  0 --> array1
   *  1 --> array2
   * @return
   */
  virtual int compare(int64_t index1, int64_t index2) const = 0;

  /**
   * equal_to of two indices.
   * IMPORTANT: to uniquely identify arrays, the most significant bit of index value is encoded as,
   *  0 --> array1
   *  1 --> array2
   * @return
   */
  virtual bool equal_to(int64_t index1, int64_t index2) const = 0;

  /**
   * compare indices of arrays by explicitly passing which array, which index
   * @param array_index1
   * @param row_index1
   * @param array_index2
   * @param row_index2
   * @return
   */
  virtual int compare(int32_t array_index1, int64_t row_index1, int32_t array_index2, int64_t row_index2) const = 0;
};
/**
 * Creates a comparator for two arrays
 * IMPORTANT: to uniquely identify rows of arr1 and arr2, the most significant bit of index value is encoded as,
 *  0 --> arr1
 *  1 --> arr2
 * @param a1
 * @param a2
 * @param asc
 * @return
 */
std::shared_ptr<TwoArrayIndexComparator> CreateTwoArrayIndexComparator(const std::shared_ptr<arrow::Array> &a1,
                                                                       const std::shared_ptr<arrow::Array> &a2,
                                                                       bool asc = true);

// -----------------------------------------------------------------------------

/**
 * @deprecated
 */
class RowEqualTo {
 public:
  RowEqualTo(std::shared_ptr<cylon::CylonContext> &ctx, const std::shared_ptr<arrow::Table> *tables,
             int64_t *eq, int64_t *hs);

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
class TableRowIndexEqualTo {
 public:
  TableRowIndexEqualTo(const std::shared_ptr<arrow::Table> &table, const std::vector<int> &col_ids);

  // equality
  bool operator()(const int64_t &record1, const int64_t &record2) const;

  // equality, less than, greater than
  int compare(const int64_t &record1, const int64_t &record2) const;

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
class ArrayIndexHash {
 public:
  explicit ArrayIndexHash(const std::shared_ptr<arrow::Array> &arr);

  // hashing
  size_t operator()(const int64_t &record) const;

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  // hence they are wrapped around smart pointers
  std::shared_ptr<std::vector<uint32_t>> hashes_ptr;
};

// -----------------------------------------------------------------------------

/**
 * hash index within a table based on multiple column indices.
 * Note: A composite hash would be precomputed in the constructor
 */
class TableRowIndexHash {
 public:
  explicit TableRowIndexHash(const std::shared_ptr<arrow::Table> &table);

  TableRowIndexHash(const std::shared_ptr<arrow::Table> &table, const std::vector<int> &col_ids);

  explicit TableRowIndexHash(const std::vector<std::shared_ptr<arrow::Array>> &arrays);

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
 * Hash function for two tables, based on rows.
 * IMPORTANT: to uniquely identify rows of t1 and t2, the most significant bit of index value is encoded as,
 *  0 --> t1
 *  1 --> t2
 *
 *  Note: A composite hash would be precomputed in the constructor for both tables
 */
class TwoTableRowIndexHash {
 public:
  TwoTableRowIndexHash(const std::shared_ptr<arrow::Table> &t1, const std::shared_ptr<arrow::Table> &t2);

  TwoTableRowIndexHash(const std::shared_ptr<arrow::Table> &t1, const std::shared_ptr<arrow::Table> &t2,
                       const std::vector<int> &t1_indices, const std::vector<int> &t2_indices);

  // hashing
  size_t operator()(int64_t idx) const;

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  std::array<std::shared_ptr<TableRowIndexHash>, 2> table_hashes;
};

// -----------------------------------------------------------------------------

/**
 * Equal-to function for two tables, based on the rows.
 * IMPORTANT: to uniquely identify rows of t1 and t2, the most significant bit of index value is encoded as,
 *  0 --> t1
 *  1 --> t2
 */
class TwoTableRowIndexEqualTo {
 public:
  TwoTableRowIndexEqualTo(const std::shared_ptr<arrow::Table> &t1, const std::shared_ptr<arrow::Table> &t2);

  TwoTableRowIndexEqualTo(const std::shared_ptr<arrow::Table> &t1, const std::shared_ptr<arrow::Table> &t2,
                          const std::vector<int> &t1_indices, const std::vector<int> &t2_indices);

  // hashing
  bool operator()(const int64_t &record1, const int64_t &record2) const;

  int compare(const int64_t &record1, const int64_t &record2) const;

  int compare(const int32_t &table1,
              const int64_t &record1,
              const int32_t &table2,
              const int64_t &record2) const;

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  std::vector<std::shared_ptr<TwoArrayIndexComparator>> comparators;
};

// -----------------------------------------------------------------------------

/**
 * Hash function for two arrays.
 * IMPORTANT: to uniquely identify rows of arr1 and arr2, the most significant bit of index value is encoded as,
 *  0 --> arr1
 *  1 --> arr2
 *
 *  Note: A composite hash would be precomputed in the constructor for both arrays
 */
class TwoArrayIndexHash {
 public:
  TwoArrayIndexHash(const std::shared_ptr<arrow::Array> &arr1, const std::shared_ptr<arrow::Array> &arr2);

  // hashing
  size_t operator()(int64_t idx) const;

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  // we can use a TableRowIndexHash objects with just 1 array
  std::array<std::shared_ptr<ArrayIndexHash>, 2> array_hashes;
};

// -----------------------------------------------------------------------------

/**
 * Equal-to function for two arrays
 * IMPORTANT: to uniquely identify rows of arr1 and arr2, the most significant bit of index value is encoded as,
 *  0 --> arr1
 *  1 --> arr2
 */
class TwoArrayIndexEqualTo {
 public:
  TwoArrayIndexEqualTo(const std::shared_ptr<arrow::Array> &arr1, const std::shared_ptr<arrow::Array> &arr2);

  // hashing
  bool operator()(const int64_t &record1, const int64_t &record2) const;

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  std::shared_ptr<TwoArrayIndexComparator> comparator;
};


// -----------------------------------------------------------------------------

/**
 * multi-table comparator based on a <table index, row index> pair
 */
class MultiTableRowIndexEqualTo {
 public:
  explicit MultiTableRowIndexEqualTo(const std::vector<std::shared_ptr<arrow::Table>> &tables)
      : tables(tables), comparator(std::make_shared<TableRowComparator>(tables[0]->fields())) {}

  bool operator()(const std::pair<int8_t, int64_t> &record1, const std::pair<int8_t, int64_t> &record2) const {
    return this->comparator->compare(this->tables[record1.first], record1.second,
                                     this->tables[record2.first], record2.second) == 0;
  }

  // equality, less than, greater than
  int compare(const std::pair<int8_t, int64_t> &record1, const std::pair<int8_t, int64_t> &record2) const;

 private:
  const std::vector<std::shared_ptr<arrow::Table>> &tables;
  std::shared_ptr<TableRowComparator> comparator;
};

}  // namespace cylon

#endif //CYLON_SRC_CYLON_ARROW_ARROW_COMPARATOR_HPP_
