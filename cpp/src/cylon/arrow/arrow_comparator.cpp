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
#include "util/arrow_utils.hpp"

#include <glog/logging.h>

namespace cylon {

template<typename ARROW_TYPE>
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
    case arrow::Type::NA:break;
    case arrow::Type::BOOL:break;
    case arrow::Type::UINT8:return std::make_shared<NumericArrowComparator<arrow::UInt8Type>>();
    case arrow::Type::INT8:return std::make_shared<NumericArrowComparator<arrow::Int8Type>>();
    case arrow::Type::UINT16:return std::make_shared<NumericArrowComparator<arrow::UInt16Type>>();
    case arrow::Type::INT16:return std::make_shared<NumericArrowComparator<arrow::Int16Type>>();
    case arrow::Type::UINT32:return std::make_shared<NumericArrowComparator<arrow::UInt32Type>>();
    case arrow::Type::INT32:return std::make_shared<NumericArrowComparator<arrow::Int16Type>>();
    case arrow::Type::UINT64:return std::make_shared<NumericArrowComparator<arrow::UInt64Type>>();
    case arrow::Type::INT64:return std::make_shared<NumericArrowComparator<arrow::Int64Type>>();
    case arrow::Type::HALF_FLOAT:return std::make_shared<NumericArrowComparator<arrow::HalfFloatType>>();
    case arrow::Type::FLOAT:return std::make_shared<NumericArrowComparator<arrow::FloatType>>();
    case arrow::Type::DOUBLE:return std::make_shared<NumericArrowComparator<arrow::DoubleType>>();
    case arrow::Type::STRING:return std::make_shared<BinaryArrowComparator>();
    case arrow::Type::BINARY:return std::make_shared<BinaryArrowComparator>();
    case arrow::Type::FIXED_SIZE_BINARY:return std::make_shared<FixedSizeBinaryArrowComparator>();
    case arrow::Type::DATE32:return std::make_shared<NumericArrowComparator<arrow::Date32Type>>();
    case arrow::Type::DATE64:return std::make_shared<NumericArrowComparator<arrow::Date64Type>>();
    case arrow::Type::TIMESTAMP:return std::make_shared<NumericArrowComparator<arrow::TimestampType>>();
    case arrow::Type::TIME32:return std::make_shared<NumericArrowComparator<arrow::Time32Type>>();
    case arrow::Type::TIME64:return std::make_shared<NumericArrowComparator<arrow::Time64Type>>();
    default: break;
  }
  return nullptr;
}

TableRowComparator::TableRowComparator(const std::vector<std::shared_ptr<arrow::Field>> &fields)
    : comparators({}) {
  comparators.reserve(fields.size());
  for (const auto &field : fields) {
    std::shared_ptr<ArrowComparator> comp = GetComparator(field->type());
    if (comp == nullptr) throw "Unable to find comparator for type " + field->type()->name();
    this->comparators.push_back(std::move(comp));
  }
}

int TableRowComparator::compare(const std::shared_ptr<arrow::Table> &table1, int64_t index1,
                                const std::shared_ptr<arrow::Table> &table2, int64_t index2) {
  // not doing schema validations here due to performance overheads. Don't expect users to use
  // this function before calling this function from an internal cylon function,
  // schema validation should be done to make sure
  // table1 and table2 has the same schema.
  for (int c = 0; c < table1->num_columns(); ++c) {
    int comparision = this->comparators[c]->compare(cylon::util::GetChunkOrEmptyArray(table1->column(c), 0), index1,
                                                    cylon::util::GetChunkOrEmptyArray(table2->column(c), 0), index2);
    if (comparision) return comparision;
  }
  return 0;
}

template<typename TYPE, bool ASC>
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

  bool equal_to(const int64_t index1, const int64_t index2) const override {
    return casted_arr->Value(index1) == casted_arr->Value(index2);
  }

 private:
  std::shared_ptr<ARROW_ARRAY_T> casted_arr;
};

template<typename TYPE, bool ASC>
class BinaryRowIndexComparator : public ArrayIndexComparator {
  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;

 public:
  explicit BinaryRowIndexComparator(const std::shared_ptr<arrow::Array> &array)
      : casted_arr(std::static_pointer_cast<ARROW_ARRAY_T>(array)) {}

  int compare(int64_t index1, int64_t index2) const override {
    if (ASC) {
      return casted_arr->GetView(index1).compare(casted_arr->GetView(index2));
    } else {
      return casted_arr->GetView(index2).compare(casted_arr->GetView(index1));
    }
  }

  bool equal_to(const int64_t index1, const int64_t index2) const override {
    return casted_arr->GetView(index1).compare(casted_arr->GetView(index2));
  }

 private:
  std::shared_ptr<arrow::BinaryArray> casted_arr;
};

class EmptyIndexComparator : public ArrayIndexComparator {
 public:
  explicit EmptyIndexComparator() = default;

  int compare(int64_t index1, int64_t index2) const override { return 0; }

  bool equal_to(const int64_t index1, const int64_t index2) const override { return true; }
};

template<bool ASC>
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

  bool equal_to(const int64_t index1, const int64_t index2) const override {
    return casted_arr->GetView(index1).compare(casted_arr->GetView(index2));
  }

 private:
  std::shared_ptr<arrow::FixedSizeBinaryArray> casted_arr;
};

template<bool ASC>
std::shared_ptr<ArrayIndexComparator> CreateArrayIndexComparatorUtil(const std::shared_ptr<arrow::Array> &array) {
  switch (array->type_id()) {
    case arrow::Type::UINT8:return std::make_shared<NumericRowIndexComparator<arrow::UInt8Type, ASC>>(array);
    case arrow::Type::INT8:return std::make_shared<NumericRowIndexComparator<arrow::Int8Type, ASC>>(array);
    case arrow::Type::UINT16:return std::make_shared<NumericRowIndexComparator<arrow::UInt16Type, ASC>>(array);
    case arrow::Type::INT16:return std::make_shared<NumericRowIndexComparator<arrow::Int16Type, ASC>>(array);
    case arrow::Type::UINT32:return std::make_shared<NumericRowIndexComparator<arrow::UInt32Type, ASC>>(array);
    case arrow::Type::INT32:return std::make_shared<NumericRowIndexComparator<arrow::Int32Type, ASC>>(array);
    case arrow::Type::UINT64:return std::make_shared<NumericRowIndexComparator<arrow::UInt64Type, ASC>>(array);
    case arrow::Type::INT64:return std::make_shared<NumericRowIndexComparator<arrow::Int64Type, ASC>>(array);
    case arrow::Type::HALF_FLOAT:return std::make_shared<NumericRowIndexComparator<arrow::HalfFloatType, ASC>>(array);
    case arrow::Type::FLOAT:return std::make_shared<NumericRowIndexComparator<arrow::FloatType, ASC>>(array);
    case arrow::Type::DOUBLE:return std::make_shared<NumericRowIndexComparator<arrow::DoubleType, ASC>>(array);
    case arrow::Type::STRING:return std::make_shared<BinaryRowIndexComparator<arrow::StringType, ASC>>(array);
    case arrow::Type::BINARY:return std::make_shared<BinaryRowIndexComparator<arrow::BinaryType, ASC>>(array);
    case arrow::Type::FIXED_SIZE_BINARY:return std::make_shared<FixedSizeBinaryRowIndexComparator<ASC>>(array);
    case arrow::Type::DATE32:return std::make_shared<NumericRowIndexComparator<arrow::Date32Type, ASC>>(array);
    case arrow::Type::DATE64:return std::make_shared<NumericRowIndexComparator<arrow::Date64Type, ASC>>(array);
    case arrow::Type::TIMESTAMP:return std::make_shared<NumericRowIndexComparator<arrow::TimestampType, ASC>>(array);
    case arrow::Type::TIME32:return std::make_shared<NumericRowIndexComparator<arrow::Time32Type, ASC>>(array);
    case arrow::Type::TIME64:return std::make_shared<NumericRowIndexComparator<arrow::Time64Type, ASC>>(array);
    default:return nullptr;
  }
}

template<typename TYPE, bool ASC = true>
class TwoNumericRowIndexComparator : public TwoArrayIndexComparator {
  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;

 public:
  TwoNumericRowIndexComparator(const std::shared_ptr<arrow::Array> &a1, const std::shared_ptr<arrow::Array> &a2)
      : arrays({std::static_pointer_cast<ARROW_ARRAY_T>(a1), std::static_pointer_cast<ARROW_ARRAY_T>(a2)}) {}

  int compare(int64_t index1, int64_t index2) const override {
    auto diff = arrays.at(util::CheckBit(index1))->Value(util::ClearBit(index1))
        - arrays.at(util::CheckBit(index2))->Value(util::ClearBit(index2));
    if (ASC) {
      return (diff > 0) - (diff < 0);
    } else {
      return (diff < 0) - (diff > 0);
    }
  }

  int compare(int32_t array_index1, int64_t row_index1, int32_t array_index2, int64_t row_index2) const override {
    auto diff = arrays.at(array_index1)->Value(row_index1) - arrays.at(array_index2)->Value(row_index2);
    if (ASC) {
      return (diff > 0) - (diff < 0);
    } else {
      return (diff < 0) - (diff > 0);
    }
  }

  bool equal_to(const int64_t index1, const int64_t index2) const override {
    return arrays.at(util::CheckBit(index1))->Value(util::ClearBit(index1))
        == arrays.at(util::CheckBit(index2))->Value(util::ClearBit(index2));
  }

 private:
  std::array<std::shared_ptr<ARROW_ARRAY_T>, 2> arrays;
};

template<typename TYPE, bool ASC>
class TwoBinaryRowIndexComparator : public TwoArrayIndexComparator {
  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;
 public:
  explicit TwoBinaryRowIndexComparator(const std::shared_ptr<arrow::Array> &a1, const std::shared_ptr<arrow::Array> &a2)
      : arrays({std::static_pointer_cast<ARROW_ARRAY_T>(a1), std::static_pointer_cast<ARROW_ARRAY_T>(a1)}) {}

  int compare(int64_t index1, int64_t index2) const override {
    if (ASC) {
      return arrays.at(util::CheckBit(index1))->GetView(util::ClearBit(index1))
          .compare(arrays.at(util::CheckBit(index2))->GetView(util::ClearBit(index2)));
    } else {
      return arrays.at(util::CheckBit(index2))->GetView(util::ClearBit(index2))
          .compare(arrays.at(util::CheckBit(index1))->GetView(util::ClearBit(index1)));
    }
  }

  bool equal_to(const int64_t index1, const int64_t index2) const override {
    return arrays.at(util::CheckBit(index1))->GetView(util::ClearBit(index1))
        .compare(arrays.at(util::CheckBit(index2))->GetView(util::ClearBit(index2)));
  }

  int compare(int32_t array_index1, int64_t row_index1, int32_t array_index2, int64_t row_index2) const override {
    if (ASC) {
      return arrays.at(array_index1)->GetView(row_index1).compare(arrays.at(array_index2)->GetView(row_index2));
    } else {
      return arrays.at(array_index2)->GetView(row_index2).compare(arrays.at(array_index1)->GetView(row_index1));
    }
  }

 private:
  std::array<std::shared_ptr<ARROW_ARRAY_T>, 2> arrays;
};

template<bool ASC>
std::shared_ptr<TwoArrayIndexComparator> CreateArrayIndexComparatorUtil(const std::shared_ptr<arrow::Array> &a1,
                                                                        const std::shared_ptr<arrow::Array> &a2) {
  if (!a1->type()->Equals(a2->type())) {
    throw "array types are not equal";
  }

  switch (a1->type_id()) {
    case arrow::Type::UINT8:return std::make_shared<TwoNumericRowIndexComparator<arrow::UInt8Type, ASC>>(a1, a2);
    case arrow::Type::INT8:return std::make_shared<TwoNumericRowIndexComparator<arrow::Int8Type, ASC>>(a1, a2);
    case arrow::Type::UINT16:return std::make_shared<TwoNumericRowIndexComparator<arrow::UInt16Type, ASC>>(a1, a2);
    case arrow::Type::INT16:return std::make_shared<TwoNumericRowIndexComparator<arrow::Int16Type, ASC>>(a1, a2);
    case arrow::Type::UINT32:return std::make_shared<TwoNumericRowIndexComparator<arrow::UInt32Type, ASC>>(a1, a2);
    case arrow::Type::INT32:return std::make_shared<TwoNumericRowIndexComparator<arrow::Int32Type, ASC>>(a1, a2);
    case arrow::Type::UINT64:return std::make_shared<TwoNumericRowIndexComparator<arrow::UInt64Type, ASC>>(a1, a2);
    case arrow::Type::INT64:return std::make_shared<TwoNumericRowIndexComparator<arrow::Int64Type, ASC>>(a1, a2);
    case arrow::Type::HALF_FLOAT:
      return std::make_shared<TwoNumericRowIndexComparator<arrow::HalfFloatType, ASC>>(a1,
                                                                                       a2);
    case arrow::Type::FLOAT:return std::make_shared<TwoNumericRowIndexComparator<arrow::FloatType, ASC>>(a1, a2);
    case arrow::Type::DOUBLE:return std::make_shared<TwoNumericRowIndexComparator<arrow::DoubleType, ASC>>(a1, a2);
    case arrow::Type::STRING:
    case arrow::Type::BINARY:return std::make_shared<TwoBinaryRowIndexComparator<arrow::BinaryType, ASC>>(a1, a2);
    case arrow::Type::FIXED_SIZE_BINARY:
      return std::make_shared<TwoBinaryRowIndexComparator<arrow::FixedSizeBinaryType,
                                                          ASC>>(a1, a2);
    case arrow::Type::DATE32:return std::make_shared<TwoNumericRowIndexComparator<arrow::Date32Type, ASC>>(a1, a2);
    case arrow::Type::DATE64:return std::make_shared<TwoNumericRowIndexComparator<arrow::Date64Type, ASC>>(a1, a2);
    case arrow::Type::TIMESTAMP:
      return std::make_shared<TwoNumericRowIndexComparator<arrow::TimestampType, ASC>>(a1,
                                                                                       a2);
    case arrow::Type::TIME32:return std::make_shared<TwoNumericRowIndexComparator<arrow::Time32Type, ASC>>(a1, a2);
    case arrow::Type::TIME64:return std::make_shared<TwoNumericRowIndexComparator<arrow::Time64Type, ASC>>(a1, a2);
    default:return nullptr;
  }
}

std::shared_ptr<ArrayIndexComparator> CreateArrayIndexComparator(const std::shared_ptr<arrow::Array> &array, bool asc) {
  if (asc) {
    return CreateArrayIndexComparatorUtil<true>(array);
  } else {
    return CreateArrayIndexComparatorUtil<false>(array);
  }
}
std::shared_ptr<TwoArrayIndexComparator> CreateTwoArrayIndexComparator(const std::shared_ptr<arrow::Array> &a1,
                                                                       const std::shared_ptr<arrow::Array> &a2,
                                                                       bool asc) {
  if (asc) {
    return CreateArrayIndexComparatorUtil<true>(a1, a2);
  } else {
    return CreateArrayIndexComparatorUtil<false>(a1, a2);
  }
}

RowEqualTo::RowEqualTo(std::shared_ptr<cylon::CylonContext> &ctx,
                       const std::shared_ptr<arrow::Table> *tables, int64_t *eq,
                       int64_t *hs) {
  this->tables = tables;
  this->comparator = std::make_shared<cylon::TableRowComparator>(tables[0]->fields());
  this->row_hashing_kernel = std::make_shared<cylon::RowHashingKernel>(tables[0]->fields());
  this->eq = eq;
  this->hs = hs;
}

bool RowEqualTo::operator()(const std::pair<int8_t, int64_t> &record1,
                            const std::pair<int8_t, int64_t> &record2) const {
  (*this->eq)++;
  return this->comparator->compare(this->tables[record1.first], record1.second,
                                   this->tables[record2.first], record2.second) == 0;
}

// hashing
size_t RowEqualTo::operator()(const std::pair<int8_t, int64_t> &record) const {
  (*this->hs)++;
  size_t hash = this->row_hashing_kernel->Hash(this->tables[record.first], record.second);
  return hash;
}

TableRowIndexEqualTo::TableRowIndexEqualTo(const std::shared_ptr<arrow::Table> &table,
                                           const std::vector<int> &col_ids)
    : idx_comparators_ptr(
    std::make_shared<std::vector<std::shared_ptr<ArrayIndexComparator>>>(col_ids.size())) {
  for (size_t c = 0; c < col_ids.size(); c++) {
    if (table->column(col_ids.at(c))->num_chunks() == 0) {
      this->idx_comparators_ptr->at(c) = std::make_shared<EmptyIndexComparator>();
    } else {
      const std::shared_ptr<arrow::Array> &array = cylon::util::GetChunkOrEmptyArray(table->column(col_ids.at(c)), 0);
      this->idx_comparators_ptr->at(c) = CreateArrayIndexComparator(array);
      if (this->idx_comparators_ptr->at(c) == nullptr) {
        throw "Unable to find comparator for type " + array->type()->name();
      }
    }
  }
}
bool TableRowIndexEqualTo::operator()(const int64_t &record1, const int64_t &record2) const {
  for (auto &&comp : *idx_comparators_ptr) {
    if (comp->compare(record1, record2)) return false;
  }
  return true;
}

int TableRowIndexEqualTo::compare(const int64_t &record1, const int64_t &record2) const {
  for (auto &&comp : *idx_comparators_ptr) {
    auto res = comp->compare(record1, record2);
    if (res == 0) {
      continue;
    } else {
      return res;
    }
  }
  return 0;
}

TableRowIndexHash::TableRowIndexHash(const std::shared_ptr<arrow::Table> &table) :
    TableRowIndexHash::TableRowIndexHash(table, {}) {}

TableRowIndexHash::TableRowIndexHash(const std::shared_ptr<arrow::Table> &table, const std::vector<int> &col_ids)
    : hashes_ptr(std::make_shared<std::vector<uint32_t>>(table->num_rows(), 0)) {
  if (!col_ids.empty()) {
    for (auto &&c : col_ids) {
      const std::unique_ptr<HashPartitionKernel> &hash_kernel = CreateHashPartitionKernel(table->field(c)->type());
      if (hash_kernel == nullptr) {
        throw "Unable to find comparator for type " + table->field(c)->type()->name();
      }
      const auto &s = hash_kernel->UpdateHash(table->column(c), *hashes_ptr);  // update the hashes
      if (!s.is_ok()) {
        throw "hash update failed!";
      }
    }
  } else { // if col_ids is empty, use all columns
    for (int i = 0; i < table->num_columns(); i++) {
      const auto &hash_kernel = CreateHashPartitionKernel(table->field(i)->type());
      if (hash_kernel == nullptr) {
        throw "Unable to find comparator for type " + table->field(i)->type()->name();
      }
      const auto &s = hash_kernel->UpdateHash(table->column(i), *hashes_ptr);  // update the hashes
      if (!s.is_ok()) {
        throw "hash update failed!";
      }
    }
  }
}

TableRowIndexHash::TableRowIndexHash(const std::vector<std::shared_ptr<arrow::Array>> &arrays)
    : hashes_ptr(std::make_shared<std::vector<uint32_t>>(arrays[0]->length(), 0)) {
  for (const auto &arr: arrays) {
    const auto &hash_kernel = CreateHashPartitionKernel(arr->type());
    if (hash_kernel == nullptr) {
      throw "Unable to find comparator for type " + arr->type()->name();
    }
    const auto &s = hash_kernel->UpdateHash(arr, *hashes_ptr);  // update the hashes
    if (!s.is_ok()) {
      throw "hash update failed!";
    }
  }
}

size_t TableRowIndexHash::operator()(const int64_t &record) const { return hashes_ptr->at(record); }

std::shared_ptr<arrow::UInt32Array> TableRowIndexHash::GetHashArray(const TableRowIndexHash &hasher) {
  const auto &buf = arrow::Buffer::Wrap(*hasher.hashes_ptr);
  const auto &data =
      arrow::ArrayData::Make(arrow::uint32(), hasher.hashes_ptr->size(), {nullptr, buf});
  return std::make_shared<arrow::UInt32Array>(data);
}

TwoTableRowIndexHash::TwoTableRowIndexHash(const std::shared_ptr<arrow::Table> &t1,
                                           const std::shared_ptr<arrow::Table> &t2)
    : table_hashes({std::make_shared<TableRowIndexHash>(t1),
                    std::make_shared<TableRowIndexHash>(t2)}) {
  if (t1->num_columns() != t2->num_columns()) {
    throw "num columns of t1 and t2 are not equal!";
  }
}

TwoTableRowIndexHash::TwoTableRowIndexHash(const std::shared_ptr<arrow::Table> &t1,
                                           const std::shared_ptr<arrow::Table> &t2,
                                           const std::vector<int> &t1_indices,
                                           const std::vector<int> &t2_indices)
    : table_hashes({std::make_shared<TableRowIndexHash>(t1, t1_indices),
                    std::make_shared<TableRowIndexHash>(t2, t2_indices)}) {
  if (t1_indices.size() != t2_indices.size()) {
    throw "sizes of indices of t1 and t2 are not equal!";
  }
}

size_t TwoTableRowIndexHash::operator()(int64_t idx) const {
  return table_hashes.at(util::CheckBit(idx))->operator()(util::ClearBit(idx));
}

int MultiTableRowIndexEqualTo::compare(const std::pair<int8_t, int64_t> &record1,
                                       const std::pair<int8_t, int64_t> &record2) const {
  return this->comparator->compare(this->tables[record1.first], record1.second,
                                   this->tables[record2.first], record2.second);
}

TwoTableRowIndexEqualTo::TwoTableRowIndexEqualTo(const std::shared_ptr<arrow::Table> &t1,
                                                 const std::shared_ptr<arrow::Table> &t2,
                                                 const std::vector<int> &t1_indices,
                                                 const std::vector<int> &t2_indices)
    : comparators(t1_indices.size()) {
  if (t1_indices.size() != t2_indices.size()) {
    throw "sizes of indices of t1 and t2 are not equal!";
  }

  for (size_t i = 0; i < t1_indices.size(); i++) {
    const std::shared_ptr<arrow::Array> &a1 = util::GetChunkOrEmptyArray(t1->column(t1_indices[i]), 0);
    const std::shared_ptr<arrow::Array> &a2 = util::GetChunkOrEmptyArray(t2->column(t2_indices[i]), 0);
    comparators[i] = CreateTwoArrayIndexComparator(a1, a2);
    if (comparators[i] == nullptr) throw "Unable to find comparator for type " + a1->type()->name();
  }
}

TwoTableRowIndexEqualTo::TwoTableRowIndexEqualTo(const std::shared_ptr<arrow::Table> &t1,
                                                 const std::shared_ptr<arrow::Table> &t2)
    : comparators(t1->num_columns()) {
  if (t1->num_columns() != t2->num_columns()) {
    throw "num columns of t1 and t2 are not equal!";
  }

  for (int i = 0; i < t1->num_columns(); i++) {
    const std::shared_ptr<arrow::Array> &a1 = util::GetChunkOrEmptyArray(t1->column(i), 0);
    const std::shared_ptr<arrow::Array> &a2 = util::GetChunkOrEmptyArray(t2->column(i), 0);
    comparators[i] = CreateTwoArrayIndexComparator(a1, a2);
    if (comparators[i] == nullptr) throw "Unable to find comparator for type " + a1->type()->name();
  }
}

bool TwoTableRowIndexEqualTo::operator()(const int64_t &record1, const int64_t &record2) const {
  for (const auto &comp:comparators) {
    if (comp->compare(record1, record2)) return false;
  }
  return true;
}

int TwoTableRowIndexEqualTo::compare(const int64_t &record1, const int64_t &record2) const {
  for (const auto &comp:comparators) {
    auto com = comp->compare(record1, record2);
    if (com == 0) {
      continue;
    } else {
      return com;
    }
  }
  return 0;
}

int TwoTableRowIndexEqualTo::compare(const int32_t &table1,
                                     const int64_t &record1,
                                     const int32_t &table2,
                                     const int64_t &record2) const {
  for (const auto &comp:comparators) {
    auto com = comp->compare(table1, record1, table2, record2);
    if (com == 0) {
      continue;
    } else {
      return com;
    }
  }
  return 0;
}

TwoArrayIndexHash::TwoArrayIndexHash(const std::shared_ptr<arrow::Array> &arr1,
                                     const std::shared_ptr<arrow::Array> &arr2)
    : array_hashes({std::make_shared<ArrayIndexHash>(arr1), std::make_shared<ArrayIndexHash>(arr2)}) {}

size_t TwoArrayIndexHash::operator()(int64_t idx) const {
  return array_hashes.at(util::CheckBit(idx))->operator()(util::ClearBit(idx));
}

TwoArrayIndexEqualTo::TwoArrayIndexEqualTo(const std::shared_ptr<arrow::Array> &arr1,
                                           const std::shared_ptr<arrow::Array> &arr2)
    : comparator(CreateTwoArrayIndexComparator(arr1, arr2)) {}

bool TwoArrayIndexEqualTo::operator()(const int64_t &record1, const int64_t &record2) const {
  return comparator->compare(record1, record2) == 0;
}

ArrayIndexHash::ArrayIndexHash(const std::shared_ptr<arrow::Array> &arr)
    : hashes_ptr(std::make_shared<std::vector<uint32_t>>(arr->length(), 0)) {
  const auto &hash_kernel = CreateHashPartitionKernel(arr->type());
  if (hash_kernel == nullptr) {
    throw "Unable to find comparator for type " + arr->type()->name();
  }
  const auto &s = hash_kernel->UpdateHash(arr, *hashes_ptr);  // update the hashes
  if (!s.is_ok()) {
    throw "hash update failed!";
  }
}

size_t ArrayIndexHash::operator()(const int64_t &record) const {
  return hashes_ptr->at(record);
}
}  // namespace cylon
