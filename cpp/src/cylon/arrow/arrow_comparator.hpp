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

#include <utility>
#include "../status.hpp"
#include "arrow_partition_kernels.hpp"

namespace cylon {
class ArrowComparator {
 public:
  virtual int compare(const std::shared_ptr<arrow::Array> &array1, int64_t index1,
                      const std::shared_ptr<arrow::Array> &array2, int64_t index2) = 0;
};

std::shared_ptr<ArrowComparator> GetComparator(const std::shared_ptr<arrow::DataType> &type);

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

class ArrayIndexComparator {
 public:
  virtual int compare(int64_t index1, int64_t index2) = 0;
};

std::shared_ptr<ArrayIndexComparator> CreateArrayIndexComparator(const std::shared_ptr<arrow::Array> &array);

class RowComparator {
 private:
  const std::shared_ptr<arrow::Table> *tables;
  std::shared_ptr<cylon::TableRowComparator> comparator;
  std::shared_ptr<cylon::RowHashingKernel> row_hashing_kernel;
  int64_t *eq, *hs;

 public:
  RowComparator(std::shared_ptr<cylon::CylonContext> &ctx,
                const std::shared_ptr<arrow::Table> *tables,
                int64_t *eq,
                int64_t *hs) {
    this->tables = tables;
    this->comparator = std::make_shared<cylon::TableRowComparator>(tables[0]->fields());
    this->row_hashing_kernel = std::make_shared<cylon::RowHashingKernel>(tables[0]->fields());
    this->eq = eq;
    this->hs = hs;
  }

  // equality
  bool operator()(const std::pair<int8_t, int64_t> &record1,
                  const std::pair<int8_t, int64_t> &record2) const {
    (*this->eq)++;
    return this->comparator->compare(this->tables[record1.first], record1.second,
                                     this->tables[record2.first], record2.second) == 0;
  }

  // hashing
  size_t operator()(const std::pair<int8_t, int64_t> &record) const {
    (*this->hs)++;
    size_t hash = this->row_hashing_kernel->Hash(this->tables[record.first], record.second);
    return hash;
  }
};

class TableRowIndexComparator {
 public:
  TableRowIndexComparator(const std::shared_ptr<arrow::Table> &table, const std::vector<int> &col_ids)
      : idx_comparators_ptr(std::make_shared<std::vector<std::shared_ptr<ArrayIndexComparator>>>(col_ids.size())) {
    for (size_t c=0; c < col_ids.size(); c++) {
      const std::shared_ptr<arrow::Array> &array = table->column(col_ids.at(c))->chunk(0);
      idx_comparators_ptr->at(c) = CreateArrayIndexComparator(array);
    }
  }

  // equality
  bool operator()(const int64_t &record1, const int64_t &record2) const {
    for (auto &&comp:*idx_comparators_ptr) {
      if (comp->compare(record1, record2)) return false;
    }
    return true;
  }

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  // hence they are wrapped around smart pointers
  std::shared_ptr<std::vector<std::shared_ptr<ArrayIndexComparator>>> idx_comparators_ptr;
};

class TableRowIndexHash {
 public:
  TableRowIndexHash(const std::shared_ptr<arrow::Table> &table, const std::vector<int> &col_ids)
      : hashes_ptr(std::make_shared<std::vector<uint32_t>>(table->num_rows(), 0)) {
    for (auto &&c:col_ids) {
      const std::unique_ptr<HashPartitionKernel> &hash_kernel = CreateHashPartitionKernel(table->field(c)->type());
      hash_kernel->UpdateHash(table->column(c), *hashes_ptr); // update the hashes
    }
  }

  // hashing
  size_t operator()(const int64_t &record) const {
    return hashes_ptr->at(record);
  }

  static std::shared_ptr<arrow::UInt32Array> GetHashArray(const TableRowIndexHash &hasher) {
    const auto &buf = arrow::Buffer::Wrap(*hasher.hashes_ptr);
    const auto &data = arrow::ArrayData::Make(arrow::uint32(), hasher.hashes_ptr->size(), {nullptr, buf});
    return std::make_shared<arrow::UInt32Array>(data);
  }

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  // hence they are wrapped around smart pointers
  std::shared_ptr<std::vector<uint32_t>> hashes_ptr;
};

class MultiTableRowIndexHash {
 public:
  explicit MultiTableRowIndexHash(const std::vector<std::shared_ptr<arrow::Table>> &tables)
      : hashes_ptr(std::make_shared<std::vector<TableRowIndexHash>>()) {
    for (const auto &t:tables) {
      std::vector<int> cols(t->num_columns());
      std::iota(cols.begin(), cols.end(), 0);
      hashes_ptr->emplace_back(t, cols);
    }
  }

  // hashing
  size_t operator()(const std::pair<int8_t, int64_t> &record) const {
    return hashes_ptr->at(record.first)(record.second);
  }

 private:
  // this class gets copied to std container, so we don't want to copy these vectors.
  std::shared_ptr<std::vector<TableRowIndexHash>> hashes_ptr;
};

class MultiTableRowIndexComparator {
 public:
  explicit MultiTableRowIndexComparator(const std::vector<std::shared_ptr<arrow::Table>> &tables)
      : tables(tables), comparator(std::make_shared<TableRowComparator>(tables[0]->fields())) {}

  bool operator()(const std::pair<int8_t, int64_t> &record1, const std::pair<int8_t, int64_t> &record2) const {
    return this->comparator->compare(this->tables[record1.first], record1.second,
                                     this->tables[record2.first], record2.second) == 0;
  }

 private:
  const std::vector<std::shared_ptr<arrow::Table>> &tables;
  std::shared_ptr<TableRowComparator> comparator;
};

class PartialTableRowComparator {
 private:
  std::vector<std::shared_ptr<ArrowComparator>> comparators;
  std::vector<int> columns;
 public:
  explicit PartialTableRowComparator(const std::vector<std::shared_ptr<arrow::Field>> &vector,
                                     const std::vector<int> &cols);
  int compare(const std::shared_ptr<arrow::Table> &table1,
              int64_t index1,
              const std::shared_ptr<arrow::Table> &table2,
              int64_t index2);
};

}  // namespace cylon

#endif //CYLON_SRC_CYLON_ARROW_ARROW_COMPARATOR_HPP_
