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

#include <cylon/ctx/cylon_context.hpp>
#include <cylon/arrow/arrow_partition_kernels.hpp>

namespace cylon {

/**
 * @deprecated
 */
class ArrowComparator {
 public:
  virtual ~ArrowComparator() = default;
  virtual int compare(const std::shared_ptr<arrow::Array> &array1, int64_t index1,
                      const std::shared_ptr<arrow::Array> &array2, int64_t index2) = 0;
};

std::shared_ptr<ArrowComparator> GetComparator(const std::shared_ptr<arrow::DataType> &type);

// -----------------------------------------------------------------------------

/**
 * This Util class can be used to compare the equality  of rows of two tables. This class
 * expect Tables to have just a single chunk in all its columns
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
  virtual  ~ArrayIndexComparator() = default;

  /**
   * compares indices i and j
   * @param i
   * @param j
   * @return Output is semantically similar to the following lambda.

   auto dummy_comp = [](bool null_order, bool asc, int64_t i, int64_t j) -> int {
      T v1 = arr.IsNull(i) ? (null_order ? T_MAX_VALUE : T_MIN_VALUE) : arr[i];
      T v2 = arr.IsNull(j) ? (null_order ? T_MAX_VALUE : T_MIN_VALUE) : arr[j];
      return  asc? v1.compare(v2) : v2.compare(v1);
    };
   */
  virtual int compare(const int64_t &i, const int64_t &j) const = 0;

  /**
   * checks if indices i and j are equal
   * @param i
   * @param j
   * @return Output is semantically similar to the following lambda

   auto dummy_equal = [](int64_t i, int64_t j) -> bool {
      T v1 = arr.IsNull(i) ? T_MAX_VALUE : arr[i];
      T v2 = arr.IsNull(j) ? T_MAX_VALUE : arr[j];
      return v1 == v2;
    };
   */
  virtual bool equal_to(const int64_t &i, const int64_t &j) const = 0;
};

/**
 * Creates a comparator for a single array
 * @param array
 * @param asc ? ascending: descending
 * @param null_order ? null values considered as largest : null values considered as smallest
 * @return
 */
Status CreateArrayIndexComparator(const std::shared_ptr<arrow::Array> &array,
                                  std::unique_ptr<ArrayIndexComparator> *out_comp,
                                  bool asc = true, bool null_order = true);

// -----------------------------------------------------------------------------

/**
 * To compare indices in two arrays
 */
class DualArrayIndexComparator {
 public:
  virtual  ~DualArrayIndexComparator() = default;

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
 * @param asc ? ascending: descending
 * @param null_order ? null values considered as largest : null values considered as smallest
 * @return
 */
Status CreateDualArrayIndexComparator(const std::shared_ptr<arrow::Array> &a1,
                                      const std::shared_ptr<arrow::Array> &a2,
                                      std::unique_ptr<DualArrayIndexComparator> *out_comp,
                                      bool asc = true, bool null_order = true);

// -----------------------------------------------------------------------------

/**
 * comparator to compare indices within a table based on multiple column indices
 */
class TableRowIndexEqualTo {
 public:
  explicit TableRowIndexEqualTo(
      std::shared_ptr<std::vector<std::shared_ptr<ArrayIndexComparator>>> idx_comparators_ptr)
      : comps(std::move(idx_comparators_ptr)) {}

  virtual ~TableRowIndexEqualTo() = default;

  // equality
  bool operator()(const int64_t &record1, const int64_t &record2) const;

  // equality, less than, greater than
  int compare(const int64_t &record1, const int64_t &record2) const;

  static Status Make(const std::shared_ptr<arrow::Table> &table, const std::vector<int> &col_ids,
                     std::unique_ptr<TableRowIndexEqualTo> *out_equal_to);
  static Status Make(const std::vector<std::shared_ptr<arrow::Array>> &arrays,
                     std::unique_ptr<TableRowIndexEqualTo> *out_equal_to);

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  // hence they are wrapped around smart pointers
  std::shared_ptr<std::vector<std::shared_ptr<ArrayIndexComparator>>> comps;
};

// -----------------------------------------------------------------------------

/**
 * hash index within a table based on multiple column indices.
 * Note: A composite hash would be precomputed in the constructor
 */
class TableRowIndexHash {
 public:
  explicit TableRowIndexHash(std::shared_ptr<std::vector<uint32_t>> hashes)
      : hashes_ptr(std::move(hashes)) {}

  virtual ~TableRowIndexHash() = default;

  // hashing
  size_t operator()(const int64_t &record) const { return (*hashes_ptr)[record]; }

  /**
   * Get the composite hashes as arrow::Array
   * @param hasher
   * @return
   */
  static std::shared_ptr<arrow::UInt32Array> GetHashArray(const TableRowIndexHash &hasher);

  static Status Make(const std::shared_ptr<arrow::Table> &table,
                     std::unique_ptr<TableRowIndexHash> *hash);

  static Status Make(const std::shared_ptr<arrow::Table> &table, const std::vector<int> &col_ids,
                     std::unique_ptr<TableRowIndexHash> *hash);

  static Status Make(const std::vector<std::shared_ptr<arrow::Array>> &arrays,
                     std::unique_ptr<TableRowIndexHash> *hash);

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
class DualTableRowIndexHash {
 public:
  DualTableRowIndexHash(std::unique_ptr<TableRowIndexHash> h1,
                        std::unique_ptr<TableRowIndexHash> h2)
      : table_hashes({std::move(h1), std::move(h2)}) {}

  // hashing
  size_t operator()(int64_t idx) const;

  static Status Make(const std::shared_ptr<arrow::Table> &t1,
                     const std::shared_ptr<arrow::Table> &t2,
                     std::unique_ptr<DualTableRowIndexHash> *out_hash);

  static Status Make(const std::shared_ptr<arrow::Table> &t1,
                     const std::shared_ptr<arrow::Table> &t2,
                     const std::vector<int> &t1_indices,
                     const std::vector<int> &t2_indices,
                     std::unique_ptr<DualTableRowIndexHash> *out_hash);

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
class DualTableRowIndexEqualTo {
 public:
  explicit DualTableRowIndexEqualTo(
      std::shared_ptr<std::vector<std::shared_ptr<DualArrayIndexComparator>>> comparators)
      : comparators(std::move(comparators)) {}

  static Status Make(const std::shared_ptr<arrow::Table> &t1,
                     const std::shared_ptr<arrow::Table> &t2,
                     std::unique_ptr<DualTableRowIndexEqualTo> *out_equal_to);

  static Status Make(const std::shared_ptr<arrow::Table> &t1,
                     const std::shared_ptr<arrow::Table> &t2,
                     const std::vector<int> &t1_indices,
                     const std::vector<int> &t2_indices,
                     std::unique_ptr<DualTableRowIndexEqualTo> *out_equal_to);

  bool operator()(const int64_t &record1, const int64_t &record2) const;

  int compare(const int64_t &record1, const int64_t &record2) const;

  int compare(const int32_t &table1,
              const int64_t &record1,
              const int32_t &table2,
              const int64_t &record2) const;

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  std::shared_ptr<std::vector<std::shared_ptr<DualArrayIndexComparator>>> comparators;
};

// -----------------------------------------------------------------------------

/**
 * hash index within a table based on multiple column indices.
 * Note: A composite hash would be precomputed in the constructor
 */
class ArrayIndexHash {
 public:
  explicit ArrayIndexHash(std::shared_ptr<std::vector<uint32_t>> hashes_ptr)
      : hashes_ptr(std::move(hashes_ptr)) {}

  // hashing
  size_t operator()(const int64_t &record) const;

  static Status Make(const std::shared_ptr<arrow::Array> &arr,
                     std::unique_ptr<ArrayIndexHash> *hash);

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  // hence they are wrapped around smart pointers
  std::shared_ptr<std::vector<uint32_t>> hashes_ptr;
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
class DualArrayIndexHash {
 public:
  DualArrayIndexHash(std::unique_ptr<ArrayIndexHash> h1, std::unique_ptr<ArrayIndexHash> h2)
      : array_hashes({std::move(h1), std::move(h2)}) {}

  // hashing
  size_t operator()(int64_t idx) const;

  void ClearData(size_t idx) {
    array_hashes[idx].reset();
  }

  static Status Make(const std::shared_ptr<arrow::Array> &arr1,
                     const std::shared_ptr<arrow::Array> &arr2,
                     std::unique_ptr<DualArrayIndexHash> *hash);

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
class DualArrayIndexEqualTo {
 public:
  explicit DualArrayIndexEqualTo(std::unique_ptr<DualArrayIndexComparator> comparator)
      : comparator(std::move(comparator)) {}

  // hashing
  bool operator()(const int64_t &record1, const int64_t &record2) const {
    return comparator->equal_to(record1, record2);
  }

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  std::shared_ptr<DualArrayIndexComparator> comparator;
};


}  // namespace cylon

#endif //CYLON_SRC_CYLON_ARROW_ARROW_COMPARATOR_HPP_
