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

#include "arrow_comparator.hpp"

#include <glog/logging.h>

namespace cylon {

template <typename ARROW_TYPE>
class NumericArrowComparator : public ArrowComparator {
  int compare(const std::shared_ptr<arrow::Array> &array1, int64_t index1,
              const std::shared_ptr<arrow::Array> &array2, int64_t index2) override {
    auto reader1 = std::static_pointer_cast<arrow::NumericArray<ARROW_TYPE>>(array1);
    auto reader2 = std::static_pointer_cast<arrow::NumericArray<ARROW_TYPE>>(array2);
    auto diff = reader1->Value(index1) - reader2->Value(index2);

    // source : https://graphics.stanford.edu/~seander/bithacks.html#CopyIntegerSign
    return (diff > 0) - (diff < 0);
  }
};

class BinaryArrowComparator : public ArrowComparator {
  int compare(const std::shared_ptr<arrow::Array> &array1, int64_t index1,
              const std::shared_ptr<arrow::Array> &array2, int64_t index2) override {
    auto reader1 = std::static_pointer_cast<arrow::BinaryArray>(array1);
    auto reader2 = std::static_pointer_cast<arrow::BinaryArray>(array2);

    return reader1->GetString(index1).compare(reader2->GetString(index2));
  }
};

class FixedSizeBinaryArrowComparator : public ArrowComparator {
  int compare(const std::shared_ptr<arrow::Array> &array1, int64_t index1,
              const std::shared_ptr<arrow::Array> &array2, int64_t index2) override {
    auto reader1 = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(array1);
    auto reader2 = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(array2);

    return reader1->GetString(index1).compare(reader2->GetString(index2));
  }
};

std::shared_ptr<ArrowComparator> GetComparator(const std::shared_ptr<arrow::DataType> &type) {
  switch (type->id()) {
    case arrow::Type::NA:
      break;
    case arrow::Type::BOOL:
      break;
    case arrow::Type::UINT8:
      return std::make_shared<NumericArrowComparator<arrow::UInt8Type>>();
    case arrow::Type::INT8:
      return std::make_shared<NumericArrowComparator<arrow::Int8Type>>();
    case arrow::Type::UINT16:
      return std::make_shared<NumericArrowComparator<arrow::UInt16Type>>();
    case arrow::Type::INT16:
      return std::make_shared<NumericArrowComparator<arrow::Int16Type>>();
    case arrow::Type::UINT32:
      return std::make_shared<NumericArrowComparator<arrow::UInt32Type>>();
    case arrow::Type::INT32:
      return std::make_shared<NumericArrowComparator<arrow::Int16Type>>();
    case arrow::Type::UINT64:
      return std::make_shared<NumericArrowComparator<arrow::UInt64Type>>();
    case arrow::Type::INT64:
      return std::make_shared<NumericArrowComparator<arrow::Int64Type>>();
    case arrow::Type::HALF_FLOAT:
      return std::make_shared<NumericArrowComparator<arrow::HalfFloatType>>();
    case arrow::Type::FLOAT:
      return std::make_shared<NumericArrowComparator<arrow::FloatType>>();
    case arrow::Type::DOUBLE:
      return std::make_shared<NumericArrowComparator<arrow::DoubleType>>();
    case arrow::Type::STRING:
      return std::make_shared<BinaryArrowComparator>();
    case arrow::Type::BINARY:
      return std::make_shared<BinaryArrowComparator>();
    case arrow::Type::FIXED_SIZE_BINARY:
      return std::make_shared<FixedSizeBinaryArrowComparator>();
    case arrow::Type::DATE32:
      break;
    case arrow::Type::DATE64:
      break;
    case arrow::Type::TIMESTAMP:
      break;
    case arrow::Type::TIME32:
      break;
    case arrow::Type::TIME64:
      break;
    case arrow::Type::DECIMAL:
      break;
    case arrow::Type::LIST:
      break;
    case arrow::Type::STRUCT:
      break;
    case arrow::Type::DICTIONARY:
      break;
    case arrow::Type::MAP:
      break;
    case arrow::Type::EXTENSION:
      break;
    case arrow::Type::FIXED_SIZE_LIST:
      break;
    case arrow::Type::DURATION:
      break;
    case arrow::Type::LARGE_STRING:
      break;
    case arrow::Type::LARGE_BINARY:
      break;
    case arrow::Type::LARGE_LIST:
      break;
    case arrow::Type::INTERVAL_MONTHS:
      break;
    case arrow::Type::INTERVAL_DAY_TIME:
      break;
    case arrow::Type::SPARSE_UNION:
      break;
    case arrow::Type::DENSE_UNION:
      break;
    case arrow::Type::MAX_ID:
      break;
  }
  return nullptr;
}

TableRowComparator::TableRowComparator(const std::vector<std::shared_ptr<arrow::Field>> &fields) {
  for (const auto &field : fields) {
    this->comparators.push_back(GetComparator(field->type()));
  }
}

int TableRowComparator::compare(const std::shared_ptr<arrow::Table> &table1, int64_t index1,
                                const std::shared_ptr<arrow::Table> &table2, int64_t index2) {
  // not doing schema validations here due to performance overheads. Don't expect users to use
  // this function before calling this function from an internal cylon function,
  // schema validation should be done to make sure
  // table1 and table2 has the same schema.
  for (int c = 0; c < table1->num_columns(); ++c) {
    int comparision = this->comparators[c]->compare(table1->column(c)->chunk(0), index1,
                                                    table2->column(c)->chunk(0), index2);
    if (comparision) return comparision;
  }
  return 0;
}

template <typename TYPE, bool ASC>
class NumericRowIndexComparator : public ArrayIndexComparator {
  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;

 public:
  explicit NumericRowIndexComparator(const std::shared_ptr<arrow::Array> &array)
      : casted_arr(std::static_pointer_cast<ARROW_ARRAY_T>(array)) {}

  int compare(int64_t index1, int64_t index2) const override {
    auto diff = (casted_arr->Value(index1) - casted_arr->Value(index2));
    if (ASC) {
      return (diff > 0) - (diff < 0);
    } else {
      return (diff < 0) - (diff > 0);
    }
  }

 private:
  std::shared_ptr<ARROW_ARRAY_T> casted_arr;
};

template <bool ASC>
class BinaryRowIndexComparator : public ArrayIndexComparator {
 public:
  explicit BinaryRowIndexComparator(const std::shared_ptr<arrow::Array> &array)
      : casted_arr(std::static_pointer_cast<arrow::BinaryArray>(array)) {}

  int compare(int64_t index1, int64_t index2) const override {
    if (ASC) {
      return casted_arr->GetView(index1).compare(casted_arr->GetView(index2));
    } else {
      return casted_arr->GetView(index2).compare(casted_arr->GetView(index1));
    }
  }

 private:
  std::shared_ptr<arrow::BinaryArray> casted_arr;
};

class EmptyIndexComparator : public ArrayIndexComparator {
 public:
  explicit EmptyIndexComparator() {}

  int compare(int64_t index1, int64_t index2) const override { return 0; }
};

template <bool ASC>
class FixedSizeBinaryRowIndexComparator : public ArrayIndexComparator {
 public:
  explicit FixedSizeBinaryRowIndexComparator(const std::shared_ptr<arrow::Array> &array)
      : casted_arr(std::static_pointer_cast<arrow::FixedSizeBinaryArray>(array)) {}

  int compare(int64_t index1, int64_t index2) const override {
    if (ASC) {
      return casted_arr->GetView(index1).compare(casted_arr->GetView(index2));
    } else {
      return casted_arr->GetView(index2).compare(casted_arr->GetView(index1));
    }
  }

 private:
  std::shared_ptr<arrow::FixedSizeBinaryArray> casted_arr;
};

template <bool ASC>
std::shared_ptr<ArrayIndexComparator> CreateArrayIndexComparatorUtil(
    const std::shared_ptr<arrow::Array> &array) {
  switch (array->type_id()) {
    case arrow::Type::UINT8:
      return std::make_shared<NumericRowIndexComparator<arrow::UInt8Type, ASC>>(array);
    case arrow::Type::INT8:
      return std::make_shared<NumericRowIndexComparator<arrow::Int8Type, ASC>>(array);
    case arrow::Type::UINT16:
      return std::make_shared<NumericRowIndexComparator<arrow::UInt16Type, ASC>>(array);
    case arrow::Type::INT16:
      return std::make_shared<NumericRowIndexComparator<arrow::Int16Type, ASC>>(array);
    case arrow::Type::UINT32:
      return std::make_shared<NumericRowIndexComparator<arrow::UInt32Type, ASC>>(array);
    case arrow::Type::INT32:
      return std::make_shared<NumericRowIndexComparator<arrow::Int16Type, ASC>>(array);
    case arrow::Type::UINT64:
      return std::make_shared<NumericRowIndexComparator<arrow::UInt64Type, ASC>>(array);
    case arrow::Type::INT64:
      return std::make_shared<NumericRowIndexComparator<arrow::Int64Type, ASC>>(array);
    case arrow::Type::HALF_FLOAT:
      return std::make_shared<NumericRowIndexComparator<arrow::HalfFloatType, ASC>>(array);
    case arrow::Type::FLOAT:
      return std::make_shared<NumericRowIndexComparator<arrow::FloatType, ASC>>(array);
    case arrow::Type::DOUBLE:
      return std::make_shared<NumericRowIndexComparator<arrow::DoubleType, ASC>>(array);
    case arrow::Type::STRING:
      return std::make_shared<BinaryRowIndexComparator<ASC>>(array);
    case arrow::Type::BINARY:
      return std::make_shared<BinaryRowIndexComparator<ASC>>(array);
    case arrow::Type::FIXED_SIZE_BINARY:
      return std::make_shared<FixedSizeBinaryRowIndexComparator<ASC>>(array);
    default:
      return nullptr;
  }
}

std::shared_ptr<ArrayIndexComparator> CreateArrayIndexComparator(
    const std::shared_ptr<arrow::Array> &array, bool asc) {
  if (asc) {
    return CreateArrayIndexComparatorUtil<true>(array);
  } else {
    return CreateArrayIndexComparatorUtil<false>(array);
  }
}

RowComparator::RowComparator(std::shared_ptr<cylon::CylonContext> &ctx,
                             const std::shared_ptr<arrow::Table> *tables, int64_t *eq,
                             int64_t *hs) {
  this->tables = tables;
  this->comparator = std::make_shared<cylon::TableRowComparator>(tables[0]->fields());
  this->row_hashing_kernel = std::make_shared<cylon::RowHashingKernel>(tables[0]->fields());
  this->eq = eq;
  this->hs = hs;
}

bool RowComparator::operator()(const std::pair<int8_t, int64_t> &record1,
                               const std::pair<int8_t, int64_t> &record2) const {
  (*this->eq)++;
  return this->comparator->compare(this->tables[record1.first], record1.second,
                                   this->tables[record2.first], record2.second) == 0;
}

// hashing
size_t RowComparator::operator()(const std::pair<int8_t, int64_t> &record) const {
  (*this->hs)++;
  size_t hash = this->row_hashing_kernel->Hash(this->tables[record.first], record.second);
  return hash;
}

TableRowIndexComparator::TableRowIndexComparator(const std::shared_ptr<arrow::Table> &table,
                                                 const std::vector<int> &col_ids)
    : idx_comparators_ptr(
          std::make_shared<std::vector<std::shared_ptr<ArrayIndexComparator>>>(col_ids.size())) {
  for (size_t c = 0; c < col_ids.size(); c++) {
    if (table->column(col_ids.at(c))->num_chunks() == 0) {
      this->idx_comparators_ptr->at(c) = std::make_shared<EmptyIndexComparator>();
    } else {
      const std::shared_ptr<arrow::Array> &array = table->column(col_ids.at(c))->chunk(0);
      this->idx_comparators_ptr->at(c) = CreateArrayIndexComparator(array);
    }
  }
}
bool TableRowIndexComparator::operator()(const int64_t &record1, const int64_t &record2) const {
  for (auto &&comp : *idx_comparators_ptr) {
    if (comp->compare(record1, record2)) return false;
  }
  return true;
}

TableRowIndexHash::TableRowIndexHash(const std::shared_ptr<arrow::Table> &table,
                                     const std::vector<int> &col_ids)
    : hashes_ptr(std::make_shared<std::vector<uint32_t>>(table->num_rows(), 0)) {
  for (auto &&c : col_ids) {
    const std::unique_ptr<HashPartitionKernel> &hash_kernel =
        CreateHashPartitionKernel(table->field(c)->type());
    hash_kernel->UpdateHash(table->column(c), *hashes_ptr);  // update the hashes
  }
}
size_t TableRowIndexHash::operator()(const int64_t &record) const { return hashes_ptr->at(record); }

std::shared_ptr<arrow::UInt32Array> TableRowIndexHash::GetHashArray(
    const TableRowIndexHash &hasher) {
  const auto &buf = arrow::Buffer::Wrap(*hasher.hashes_ptr);
  const auto &data =
      arrow::ArrayData::Make(arrow::uint32(), hasher.hashes_ptr->size(), {nullptr, buf});
  return std::make_shared<arrow::UInt32Array>(data);
}

MultiTableRowIndexHash::MultiTableRowIndexHash(
    const std::vector<std::shared_ptr<arrow::Table>> &tables)
    : hashes_ptr(std::make_shared<std::vector<TableRowIndexHash>>()) {
  for (const auto &t : tables) {
    std::vector<int> cols(t->num_columns());
    std::iota(cols.begin(), cols.end(), 0);
    hashes_ptr->emplace_back(t, cols);
  }
}
size_t MultiTableRowIndexHash::operator()(const std::pair<int8_t, int64_t> &record) const {
  return hashes_ptr->at(record.first)(record.second);
}

MultiTableRowIndexComparator::MultiTableRowIndexComparator(
    const std::vector<std::shared_ptr<arrow::Table>> &tables)
    : tables(tables), comparator(std::make_shared<TableRowComparator>(tables[0]->fields())) {}
bool MultiTableRowIndexComparator::operator()(const std::pair<int8_t, int64_t> &record1,
                                              const std::pair<int8_t, int64_t> &record2) const {
  return this->comparator->compare(this->tables[record1.first], record1.second,
                                   this->tables[record2.first], record2.second) == 0;
}

}  // namespace cylon
