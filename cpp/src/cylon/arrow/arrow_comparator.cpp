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
#include "cylon/util/arrow_utils.hpp"
#include "cylon/util/macros.hpp"

#include "cylon/arrow/arrow_comparator.hpp"
#include "cylon/arrow/arrow_type_traits.hpp"

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
    case arrow::Type::NA:
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
    case arrow::Type::STRING:
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
  for (const auto &field: fields) {
    std::shared_ptr<ArrowComparator> comp = GetComparator(field->type());
    if (comp == nullptr) {
      throw std::runtime_error("Unable to find comparator for type " + field->type()->ToString());
    }
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
    int comparision = comparators[c]->compare(table1->column(c)->chunk(0), index1,
                                              table2->column(c)->chunk(0), index2);
    if (comparision != 0) {
      return comparision;
    }
  }
  return 0;
}

template<typename ArrowT, bool Asc, typename Enable = void>
struct CompareFunc {};

// compare function for integer-like values.
template<typename ArrowT, bool Asc>
struct CompareFunc<ArrowT, Asc,
                   std::enable_if_t<arrow::is_integer_type<ArrowT>::value
                                        || arrow::is_temporal_type<ArrowT>::value >> {
  using T = typename ArrowTypeTraits<ArrowT>::ValueT;
  using SignedT = typename std::make_signed<T>::type;

  // Since there are unsigned types, we need to promote diff to its signed type
  static int compare(const T &v1, const T &v2) {
    auto diff = static_cast<SignedT>(v1 - v2);
    if (Asc) {
      return (diff > 0) - (diff < 0);
    } else {
      return (diff < 0) - (diff > 0);
    }
  }
};

// compare function for float-like values
template<typename ArrowT, bool Asc>
struct CompareFunc<ArrowT, Asc, arrow::enable_if_floating_point<ArrowT>> {
  using T = typename ArrowTypeTraits<ArrowT>::ValueT;

  static int compare(const T &v1, const T &v2) {
    T diff = v1 - v2;
    if (Asc) {
      return (diff > 0) - (diff < 0);
    } else {
      return (diff < 0) - (diff > 0);
    }
  }
};

template<typename ArrowT, bool Asc>
struct CompareFunc<ArrowT, Asc, arrow::enable_if_has_string_view<ArrowT>> {

  static int compare(const arrow::util::string_view &v1, const arrow::util::string_view &v2) {
    if (Asc) {
      return v1.compare(v2);
    } else {
      return v2.compare(v1);
    }
  }
};

template<typename TYPE, bool ASC>
class NumericIndexComparator : public ArrayIndexComparator {
  using T = typename ArrowTypeTraits<TYPE>::ValueT;

 public:
  explicit NumericIndexComparator(const std::shared_ptr<arrow::Array> &array)
      : value_buffer(array->data()->template GetValues<T>(1)) {}

  int compare(const int64_t &index1, const int64_t &index2) const override {
    return CompareFunc<TYPE, ASC>::compare(value_buffer[index1], value_buffer[index2]);
  }

  bool equal_to(const int64_t &index1, const int64_t &index2) const override {
    return value_buffer[index1] == value_buffer[index2];
  }

 private:
  const T *value_buffer;
};

template<typename TYPE, bool ASC>
class BinaryRowIndexComparator : public ArrayIndexComparator {
  using ArrayType = typename arrow::TypeTraits<TYPE>::ArrayType;

 public:
  explicit BinaryRowIndexComparator(const std::shared_ptr<arrow::Array> &array)
      : casted_arr(std::static_pointer_cast<ArrayType>(array)) {}

  int compare(const int64_t &index1, const int64_t &index2) const override {
    return CompareFunc<TYPE, ASC>::compare(casted_arr->GetView(index1),
                                           casted_arr->GetView(index2));
  }

  bool equal_to(const int64_t &index1, const int64_t &index2) const override {
    return casted_arr->GetView(index1) == casted_arr->GetView(index2);
  }

 private:
  std::shared_ptr<ArrayType> casted_arr;
};

class EmptyIndexComparator : public ArrayIndexComparator {
 public:
  explicit EmptyIndexComparator() = default;

  int compare(const int64_t &index1, const int64_t &index2) const override {
    CYLON_UNUSED(index1);
    CYLON_UNUSED(index2);
    return 0;
  }

  bool equal_to(const int64_t &index1, const int64_t &index2) const override {
    CYLON_UNUSED(index1);
    CYLON_UNUSED(index2);
    return true;
  }
};

/*
 * Single implementation for both numeric and binary comparators. array->GetView(idx) method is
 * used here. For Numeric arrays, this would translate to value by pointer offset. For binary,
 * this will take arrow::util::string_view.
 */
template<typename ArrowT, bool Asc, bool NullOrder>
class ArrayIndexComparatorWithNulls : public ArrayIndexComparator {
  using T = typename ArrowTypeTraits<ArrowT>::ValueT;
  using ArrayT = typename ArrowTypeTraits<ArrowT>::ArrayT;

 public:
  explicit ArrayIndexComparatorWithNulls(const std::shared_ptr<arrow::Array> &array)
      : array(std::static_pointer_cast<ArrayT>(array)) {}

  int compare(const int64_t &index1, const int64_t &index2) const override {
    bool is_null1 = array->IsNull(index1);
    bool is_null2 = array->IsNull(index2);

    if (is_null1 || is_null2) { // if either one is null,
      return (is_null1 && !is_null2) * ((Asc == NullOrder) - (Asc != NullOrder))
          + (!is_null1 && is_null2) * ((Asc != NullOrder) - (Asc == NullOrder));
    }

    // none of the values are null. Then use the comparison trick
    return CompareFunc<ArrowT, Asc>::compare(array->GetView(index1), array->GetView(index2));
  }

  bool equal_to(const int64_t &index1, const int64_t &index2) const override {
    bool is_null1 = array->IsNull(index1);
    bool is_null2 = array->IsNull(index2);

    return (is_null1 && is_null2) ||
        (!is_null1 && !is_null2 && (array->GetView(index1) == array->GetView(index2)));
  }

 private:
  std::shared_ptr<ArrayT> array;
};

template<typename ArrowT, typename Enable = void>
struct MakeArrayIndexComparator {};

template<typename ArrowT>
struct MakeArrayIndexComparator<ArrowT, arrow::enable_if_has_c_type<ArrowT>> {
  static Status Make(const std::shared_ptr<arrow::Array> &array,
                     std::unique_ptr<ArrayIndexComparator> *out_comp, bool asc, bool null_order) {
    if (array->null_count()) {
      if (asc) {
        if (null_order) {
          *out_comp =
              std::make_unique<ArrayIndexComparatorWithNulls<ArrowT, true, true>>(array);
        } else {
          *out_comp =
              std::make_unique<ArrayIndexComparatorWithNulls<ArrowT, true, false>>(array);
        }
      } else {
        if (null_order) {
          *out_comp =
              std::make_unique<ArrayIndexComparatorWithNulls<ArrowT, false, true>>(array);
        } else {
          *out_comp =
              std::make_unique<ArrayIndexComparatorWithNulls<ArrowT, false, false>>(array);
        }
      }
    } else {
      if (asc) {
        *out_comp = std::make_unique<NumericIndexComparator<ArrowT, true>>(array);
      } else {
        *out_comp = std::make_unique<NumericIndexComparator<ArrowT, false>>(array);
      }
    }
    return Status::OK();
  }
};

template<typename ArrowT>
struct MakeArrayIndexComparator<ArrowT, arrow::enable_if_has_string_view<ArrowT>> {
  static Status Make(const std::shared_ptr<arrow::Array> &array,
                     std::unique_ptr<ArrayIndexComparator> *out_comp, bool asc, bool null_order) {
    if (array->null_count()) {
      if (asc) {
        if (null_order) {
          *out_comp =
              std::make_unique<ArrayIndexComparatorWithNulls<ArrowT, true, true>>(array);
        } else {
          *out_comp =
              std::make_unique<ArrayIndexComparatorWithNulls<ArrowT, true, false>>(array);
        }
      } else {
        if (null_order) {
          *out_comp =
              std::make_unique<ArrayIndexComparatorWithNulls<ArrowT, false, true>>(array);
        } else {
          *out_comp =
              std::make_unique<ArrayIndexComparatorWithNulls<ArrowT, false, false>>(array);
        }
      }
    } else {
      if (asc) {
        *out_comp = std::make_unique<BinaryRowIndexComparator<ArrowT, true>>(array);
      } else {
        *out_comp = std::make_unique<BinaryRowIndexComparator<ArrowT, false>>(array);
      }
    }
    return Status::OK();
  }
};

Status CreateArrayIndexComparator(const std::shared_ptr<arrow::Array> &array,
                                  std::unique_ptr<ArrayIndexComparator> *out_comp,
                                  bool asc, bool null_order) {
  if (array->length() == 0) {
    *out_comp = std::make_unique<EmptyIndexComparator>();
    return Status::OK();
  }

  switch (array->type_id()) {
    case arrow::Type::UINT8:
      return MakeArrayIndexComparator<arrow::UInt8Type>::Make(array,
                                                              out_comp,
                                                              asc, null_order);
    case arrow::Type::INT8:
      return MakeArrayIndexComparator<arrow::Int8Type>::Make(array,
                                                             out_comp,
                                                             asc, null_order);
    case arrow::Type::UINT16:
      return MakeArrayIndexComparator<arrow::UInt16Type>::Make(array,
                                                               out_comp,
                                                               asc, null_order);
    case arrow::Type::INT16:
      return MakeArrayIndexComparator<arrow::Int16Type>::Make(array,
                                                              out_comp,
                                                              asc, null_order);
    case arrow::Type::UINT32:
      return MakeArrayIndexComparator<arrow::UInt32Type>::Make(array,
                                                               out_comp,
                                                               asc, null_order);
    case arrow::Type::INT32:
      return MakeArrayIndexComparator<arrow::Int32Type>::Make(array,
                                                              out_comp,
                                                              asc, null_order);
    case arrow::Type::UINT64:
      return MakeArrayIndexComparator<arrow::UInt64Type>::Make(array,
                                                               out_comp,
                                                               asc, null_order);
    case arrow::Type::INT64:
      return MakeArrayIndexComparator<arrow::Int64Type>::Make(array,
                                                              out_comp,
                                                              asc, null_order);
    case arrow::Type::HALF_FLOAT:
      return MakeArrayIndexComparator<arrow::HalfFloatType>::Make(array,
                                                                  out_comp,
                                                                  asc, null_order);
    case arrow::Type::FLOAT:
      return MakeArrayIndexComparator<arrow::FloatType>::Make(array,
                                                              out_comp,
                                                              asc, null_order);
    case arrow::Type::DOUBLE:
      return MakeArrayIndexComparator<arrow::DoubleType>::Make(array,
                                                               out_comp,
                                                               asc, null_order);
    case arrow::Type::STRING:
      return MakeArrayIndexComparator<arrow::StringType>::Make(array,
                                                               out_comp,
                                                               asc, null_order);
    case arrow::Type::LARGE_STRING:
      return MakeArrayIndexComparator<arrow::LargeStringType>::Make(array, out_comp,
                                                                    asc, null_order);
    case arrow::Type::BINARY:
      return MakeArrayIndexComparator<arrow::BinaryType>::Make(array,
                                                               out_comp,
                                                               asc, null_order);
    case arrow::Type::LARGE_BINARY:
      return MakeArrayIndexComparator<arrow::LargeBinaryType>::Make(array, out_comp,
                                                                    asc, null_order);
    case arrow::Type::FIXED_SIZE_BINARY:
      return MakeArrayIndexComparator<arrow::FixedSizeBinaryType>::Make(array,
                                                                        out_comp,
                                                                        asc, null_order);
    case arrow::Type::DATE32:
      return MakeArrayIndexComparator<arrow::Date32Type>::Make(array,
                                                               out_comp,
                                                               asc, null_order);
    case arrow::Type::DATE64:
      return MakeArrayIndexComparator<arrow::Date64Type>::Make(array,
                                                               out_comp,
                                                               asc, null_order);
    case arrow::Type::TIMESTAMP:
      return MakeArrayIndexComparator<arrow::TimestampType>::Make(array,
                                                                  out_comp,
                                                                  asc, null_order);
    case arrow::Type::TIME32:
      return MakeArrayIndexComparator<arrow::Time32Type>::Make(array,
                                                               out_comp,
                                                               asc, null_order);
    case arrow::Type::TIME64:
      return MakeArrayIndexComparator<arrow::Time64Type>::Make(array,
                                                               out_comp,
                                                               asc, null_order);
    default:
      return {Code::Invalid,
              "Invalid data type for ArrayIndexComparator " + array->type()->ToString()};
  }
}

template<typename TYPE, bool ASC = true>
class DualNumericRowIndexComparator : public DualArrayIndexComparator {
  using T = typename TYPE::c_type;

 public:
  DualNumericRowIndexComparator(const std::shared_ptr<arrow::Array> &a1,
                                const std::shared_ptr<arrow::Array> &a2)
      : arrays({a1->data()->template GetValues<T>(1), a2->data()->template GetValues<T>(1)}) {}

  static Status Make(const std::shared_ptr<arrow::Array> &a1,
                     const std::shared_ptr<arrow::Array> &a2,
                     std::unique_ptr<DualArrayIndexComparator> *out_comp) {
    *out_comp = std::make_unique<DualNumericRowIndexComparator<TYPE, ASC>>(a1, a2);
    return Status::OK();
  }

  int compare(int64_t index1, int64_t index2) const override {
    return CompareFunc<TYPE, ASC>::compare(arrays[util::CheckBit(index1)][util::ClearBit(index1)],
                                           arrays[util::CheckBit(index2)][util::ClearBit(index2)]);
  }

  int compare(int32_t array_index1, int64_t row_index1,
              int32_t array_index2, int64_t row_index2) const override {
    return CompareFunc<TYPE, ASC>::compare(arrays[array_index1][row_index1],
                                           arrays[array_index2][row_index2]);
  }

  bool equal_to(int64_t index1, int64_t index2) const override {
    return arrays[util::CheckBit(index1)][util::ClearBit(index1)]
        == arrays[util::CheckBit(index2)][util::ClearBit(index2)];
  }

 private:
  std::array<const T *, 2> arrays;
};

template<typename TYPE, bool ASC>
class DualBinaryRowIndexComparator : public DualArrayIndexComparator {
  using ARROW_ARRAY_T = typename arrow::TypeTraits<TYPE>::ArrayType;
 public:
  explicit DualBinaryRowIndexComparator(const std::shared_ptr<arrow::Array> &a1,
                                        const std::shared_ptr<arrow::Array> &a2)
      : arrays({std::static_pointer_cast<ARROW_ARRAY_T>(a1),
                std::static_pointer_cast<ARROW_ARRAY_T>(a2)}) {}

  static Status Make(const std::shared_ptr<arrow::Array> &a1,
                     const std::shared_ptr<arrow::Array> &a2,
                     std::unique_ptr<DualArrayIndexComparator> *out_comp) {
    *out_comp = std::make_unique<DualBinaryRowIndexComparator<TYPE, ASC>>(a1, a2);
    return Status::OK();
  }

  int compare(int64_t index1, int64_t index2) const override {
    return CompareFunc<TYPE, ASC>::compare(
        arrays[util::CheckBit(index1)]->GetView(util::ClearBit(index1)),
        arrays[util::CheckBit(index2)]->GetView(util::ClearBit(index2)));
  }

  bool equal_to(int64_t index1, int64_t index2) const override {
    return arrays[util::CheckBit(index1)]->GetView(util::ClearBit(index1))
        == arrays[util::CheckBit(index2)]->GetView(util::ClearBit(index2));
  }

  int compare(int32_t array_index1, int64_t row_index1,
              int32_t array_index2, int64_t row_index2) const override {
    return CompareFunc<TYPE, ASC>::compare(arrays[array_index1]->GetView(row_index1),
                                           arrays[array_index2]->GetView(row_index2));
  }

 private:
  std::array<std::shared_ptr<ARROW_ARRAY_T>, 2> arrays;
};

class EmptyDualArrayIndexComparator : public DualArrayIndexComparator {
 public:
  EmptyDualArrayIndexComparator() = default;
  int compare(int64_t index1, int64_t index2) const override {
    CYLON_UNUSED(index1);
    CYLON_UNUSED(index2);
    return 0;
  }

  bool equal_to(int64_t index1, int64_t index2) const override {
    CYLON_UNUSED(index1);
    CYLON_UNUSED(index2);
    return false;
  }
  int compare(int32_t array_index1, int64_t row_index1,
              int32_t array_index2, int64_t row_index2) const override {
    CYLON_UNUSED(array_index1);
    CYLON_UNUSED(row_index1);
    CYLON_UNUSED(array_index2);
    CYLON_UNUSED(row_index2);
    return 0;
  }
};

class DualArrayIndexComparatorForSingleArray : public DualArrayIndexComparator {
 public:
  explicit DualArrayIndexComparatorForSingleArray(std::unique_ptr<ArrayIndexComparator> non_empty_comp)
      : non_empty_comp(std::move(non_empty_comp)) {}

  static Status Make(const std::shared_ptr<arrow::Array> &non_empty_arr,
                     std::unique_ptr<DualArrayIndexComparator> *out_comp,
                     bool asc = true) {
    std::unique_ptr<ArrayIndexComparator> non_empty_comp;
    RETURN_CYLON_STATUS_IF_FAILED(
        CreateArrayIndexComparator(non_empty_arr, &non_empty_comp, asc,/*null_order=*/false));

    *out_comp = std::make_unique<DualArrayIndexComparatorForSingleArray>(std::move(non_empty_comp));
    return Status::OK();
  }

  int compare(int64_t index1, int64_t index2) const override {
    return non_empty_comp->compare(util::ClearBit(index1), util::ClearBit(index2));
  }

  bool equal_to(int64_t index1, int64_t index2) const override {
    return non_empty_comp->equal_to(util::ClearBit(index1), util::ClearBit(index2));
  }

  int compare(int32_t array_index1,
              int64_t row_index1,
              int32_t array_index2,
              int64_t row_index2) const override {
    CYLON_UNUSED(row_index1);
    CYLON_UNUSED(row_index2);
    return non_empty_comp->compare(array_index1, array_index2);
  }

 private:
  std::shared_ptr<ArrayIndexComparator> non_empty_comp;
};

template<bool ASC>
Status CreateArrayIndexComparatorUtil(const std::shared_ptr<arrow::Array> &a1,
                                      const std::shared_ptr<arrow::Array> &a2,
                                      std::unique_ptr<DualArrayIndexComparator> *out_comp) {
  if (!a1->type()->Equals(a2->type())) {
    return {Code::Invalid, "array types are not equal " + a1->type()->ToString() + " vs "
        + a2->type()->ToString()};
  }

  switch (a1->type_id()) {
    case arrow::Type::UINT8:
      return DualNumericRowIndexComparator<arrow::UInt8Type,
                                           ASC>::Make(a1, a2, out_comp);
    case arrow::Type::INT8:
      return DualNumericRowIndexComparator<arrow::Int8Type,
                                           ASC>::Make(a1, a2, out_comp);
    case arrow::Type::UINT16:
      return DualNumericRowIndexComparator<arrow::UInt16Type,
                                           ASC>::Make(a1, a2, out_comp);
    case arrow::Type::INT16:
      return DualNumericRowIndexComparator<arrow::Int16Type,
                                           ASC>::Make(a1, a2, out_comp);
    case arrow::Type::UINT32:
      return DualNumericRowIndexComparator<arrow::UInt32Type,
                                           ASC>::Make(a1, a2, out_comp);
    case arrow::Type::INT32:
      return DualNumericRowIndexComparator<arrow::Int32Type,
                                           ASC>::Make(a1, a2, out_comp);
    case arrow::Type::UINT64:
      return DualNumericRowIndexComparator<arrow::UInt64Type,
                                           ASC>::Make(a1, a2, out_comp);
    case arrow::Type::INT64:
      return DualNumericRowIndexComparator<arrow::Int64Type,
                                           ASC>::Make(a1, a2, out_comp);
    case arrow::Type::HALF_FLOAT:
      return DualNumericRowIndexComparator<arrow::HalfFloatType, ASC>::Make(a1,
                                                                            a2, out_comp);
    case arrow::Type::FLOAT:
      return DualNumericRowIndexComparator<arrow::FloatType,
                                           ASC>::Make(a1, a2, out_comp);
    case arrow::Type::DOUBLE:
      return DualNumericRowIndexComparator<arrow::DoubleType,
                                           ASC>::Make(a1, a2, out_comp);
    case arrow::Type::STRING:
      return DualBinaryRowIndexComparator<arrow::StringType,
                                          ASC>::Make(a1, a2, out_comp);
    case arrow::Type::BINARY:
      return DualBinaryRowIndexComparator<arrow::BinaryType,
                                          ASC>::Make(a1, a2, out_comp);
    case arrow::Type::LARGE_STRING:
      return DualBinaryRowIndexComparator<arrow::LargeStringType,
                                          ASC>::Make(a1, a2, out_comp);
    case arrow::Type::LARGE_BINARY:
      return DualBinaryRowIndexComparator<arrow::LargeBinaryType,
                                          ASC>::Make(a1, a2, out_comp);
    case arrow::Type::FIXED_SIZE_BINARY:
      return DualBinaryRowIndexComparator<arrow::FixedSizeBinaryType,
                                          ASC>::Make(a1, a2, out_comp);
    case arrow::Type::DATE32:
      return DualNumericRowIndexComparator<arrow::Date32Type,
                                           ASC>::Make(a1, a2, out_comp);
    case arrow::Type::DATE64:
      return DualNumericRowIndexComparator<arrow::Date64Type,
                                           ASC>::Make(a1, a2, out_comp);
    case arrow::Type::TIMESTAMP:
      return DualNumericRowIndexComparator<arrow::TimestampType,
                                           ASC>::Make(a1, a2, out_comp);
    case arrow::Type::TIME32:
      return DualNumericRowIndexComparator<arrow::Time32Type,
                                           ASC>::Make(a1, a2, out_comp);
    case arrow::Type::TIME64:
      return DualNumericRowIndexComparator<arrow::Time64Type,
                                           ASC>::Make(a1, a2, out_comp);
    default:
      return {Code::Invalid,
              "Invalid data type for ArrayIndexComparator " + a1->type()->ToString()};
  }
}

Status CreateDualArrayIndexComparator(const std::shared_ptr<arrow::Array> &a1,
                                      const std::shared_ptr<arrow::Array> &a2,
                                      std::unique_ptr<DualArrayIndexComparator> *out_comp,
                                      bool asc, bool null_order) {
  if (a1->length() && a2->length()) { // if both array have values
    if (asc) {
      return CreateArrayIndexComparatorUtil<true>(a1, a2, out_comp);
    } else {
      return CreateArrayIndexComparatorUtil<false>(a1, a2, out_comp);
    }
  }

  if (a1->length()) { // a1 non-empty
    return DualArrayIndexComparatorForSingleArray::Make(a1, out_comp, asc);
  }

  if (a2->length()) { // a2 non-empty
    return DualArrayIndexComparatorForSingleArray::Make(a2, out_comp, asc);
  }

  // both arrays are empty
  *out_comp = std::make_unique<EmptyDualArrayIndexComparator>();
  return Status::OK();
}

Status TableRowIndexEqualTo::Make(const std::shared_ptr<arrow::Table> &table,
                                  const std::vector<int> &col_ids,
                                  std::unique_ptr<TableRowIndexEqualTo> *out_equal_to) {
  auto comps = std::make_shared<std::vector<std::shared_ptr<ArrayIndexComparator>>>(col_ids.size());
  for (size_t c = 0; c < col_ids.size(); c++) {
    if (table->num_rows() == 0) {
      (*comps)[c] = std::make_shared<EmptyIndexComparator>();
    } else {
      const auto &array = table->column(col_ids[c])->chunk(0);
      std::unique_ptr<ArrayIndexComparator> comp;
      RETURN_CYLON_STATUS_IF_FAILED(CreateArrayIndexComparator(array, &comp));
      (*comps)[c] = std::move(comp);
    }
  }

  *out_equal_to = std::make_unique<TableRowIndexEqualTo>(std::move(comps));
  return Status::OK();
}

bool TableRowIndexEqualTo::operator()(const int64_t &record1, const int64_t &record2) const {
  return std::all_of(comps->begin(), comps->end(),
                     [&](const std::shared_ptr<ArrayIndexComparator> &comp) {
                       return comp->equal_to(record1, record2);
                     });
}

int TableRowIndexEqualTo::compare(const int64_t &record1, const int64_t &record2) const {
  for (const auto &comp: *comps) {
    int res = comp->compare(record1, record2);
    if (res != 0) {
      return res;
    }
  }
  return 0;
}

class EmptyTableRowIndexHash : public TableRowIndexHash {
 public:
  explicit EmptyTableRowIndexHash() : TableRowIndexHash(nullptr) {}

  size_t operator()(const int64_t &record) const {
    CYLON_UNUSED(record);
    return 0;
  }
};

Status TableRowIndexHash::Make(const std::shared_ptr<arrow::Table> &table,
                               std::unique_ptr<TableRowIndexHash> *hash) {
  std::vector<int> col_ids(table->num_columns());
  std::iota(col_ids.begin(), col_ids.end(), 0);

  return TableRowIndexHash::Make(table, col_ids, hash);
}

Status TableRowIndexHash::Make(const std::shared_ptr<arrow::Table> &table,
                               const std::vector<int> &col_ids,
                               std::unique_ptr<TableRowIndexHash> *hash) {
  const int64_t len = table->num_rows();

  if (len == 0) {
    *hash = std::make_unique<EmptyTableRowIndexHash>();
    return Status::OK();
  }

  auto hashes_ptr = std::make_shared<std::vector<uint32_t>>(len, 0);
  for (int c: col_ids) {
    std::unique_ptr<HashPartitionKernel> kernel;
    RETURN_CYLON_STATUS_IF_FAILED(CreateHashPartitionKernel(table->field(c)->type(), &kernel));
    RETURN_CYLON_STATUS_IF_FAILED(kernel->UpdateHash(table->column(c), *hashes_ptr));
  }
  *hash = std::make_unique<TableRowIndexHash>(std::move(hashes_ptr));
  return Status::OK();
}

Status TableRowIndexHash::Make(const std::vector<std::shared_ptr<arrow::Array>> &arrays,
                               std::unique_ptr<TableRowIndexHash> *hash) {
  const int64_t len = arrays[0]->length();
  if (std::all_of(arrays.begin() + 1, arrays.end(), [&](const std::shared_ptr<arrow::Array> &arr) {
    return arr->length() == len;
  })) {
    return {Code::Invalid, "array lengths should be equal"};
  }

  auto hashes_ptr = std::make_shared<std::vector<uint32_t>>(len, 0);
  for (const auto &arr: arrays) {
    std::unique_ptr<HashPartitionKernel> kernel;
    RETURN_CYLON_STATUS_IF_FAILED(CreateHashPartitionKernel(arr->type(), &kernel));
    RETURN_CYLON_STATUS_IF_FAILED(kernel->UpdateHash(arr, *hashes_ptr));
  }

  *hash = std::make_unique<TableRowIndexHash>(std::move(hashes_ptr));
  return Status::OK();
}

std::shared_ptr<arrow::UInt32Array> TableRowIndexHash::GetHashArray(const TableRowIndexHash &hasher) {
  auto buf = arrow::Buffer::Wrap(*hasher.hashes_ptr);
  auto data = arrow::ArrayData::Make(arrow::uint32(),
                                     static_cast<int64_t>(hasher.hashes_ptr->size()),
                                     {nullptr, std::move(buf)});
  return std::make_shared<arrow::UInt32Array>(std::move(data));
}

Status DualTableRowIndexHash::Make(const std::shared_ptr<arrow::Table> &t1,
                                   const std::shared_ptr<arrow::Table> &t2,
                                   std::unique_ptr<DualTableRowIndexHash> *out_hash) {
  if (t1->num_columns() != t2->num_columns()) {
    return {Code::Invalid, "num columns of t1 and t2 are not equal!"};
  }

  std::vector<int> col_ids(t1->num_columns());
  std::iota(col_ids.begin(), col_ids.end(), 0);

  return DualTableRowIndexHash::Make(t1, t2, col_ids, col_ids, out_hash);
}

Status DualTableRowIndexHash::Make(const std::shared_ptr<arrow::Table> &t1,
                                   const std::shared_ptr<arrow::Table> &t2,
                                   const std::vector<int> &t1_indices,
                                   const std::vector<int> &t2_indices,
                                   std::unique_ptr<DualTableRowIndexHash> *out_hash) {
  if (t1->num_columns() != t2->num_columns()) {
    return {Code::Invalid, "num columns of t1 and t2 are not equal!"};
  }

  std::unique_ptr<TableRowIndexHash> h1, h2;
  RETURN_CYLON_STATUS_IF_FAILED(TableRowIndexHash::Make(t1, t1_indices, &h1));
  RETURN_CYLON_STATUS_IF_FAILED(TableRowIndexHash::Make(t2, t2_indices, &h2));
  *out_hash = std::make_unique<DualTableRowIndexHash>(std::move(h1), std::move(h2));

  return Status::OK();
}

size_t DualTableRowIndexHash::operator()(int64_t idx) const {
  return table_hashes[util::CheckBit(idx)]->operator()(util::ClearBit(idx));
}

Status DualTableRowIndexEqualTo::Make(const std::shared_ptr<arrow::Table> &t1,
                                      const std::shared_ptr<arrow::Table> &t2,
                                      const std::vector<int> &t1_indices,
                                      const std::vector<int> &t2_indices,
                                      std::unique_ptr<DualTableRowIndexEqualTo> *out_equal_to) {
  size_t num_cols = t1_indices.size();
  if (num_cols != t2_indices.size()) {
    return {Code::Invalid, "sizes of indices of t1 and t2 are not equal!"};
  }

  auto comps = std::make_shared<std::vector<std::shared_ptr<DualArrayIndexComparator>>>(num_cols);

  for (size_t i = 0; i < num_cols; i++) {
    const auto &a1 = util::GetChunkOrEmptyArray(t1->column(t1_indices[i]), 0);
    const auto &a2 = util::GetChunkOrEmptyArray(t2->column(t2_indices[i]), 0);

    std::unique_ptr<DualArrayIndexComparator> comp;
    RETURN_CYLON_STATUS_IF_FAILED(CreateDualArrayIndexComparator(a1, a2, &comp));
    (*comps)[i] = std::move(comp);
  }

  *out_equal_to = std::make_unique<DualTableRowIndexEqualTo>(std::move(comps));
  return Status::OK();
}

Status DualTableRowIndexEqualTo::Make(const std::shared_ptr<arrow::Table> &t1,
                                      const std::shared_ptr<arrow::Table> &t2,
                                      std::unique_ptr<DualTableRowIndexEqualTo> *out_equal_to) {
  if (t1->num_columns() != t2->num_columns()) {
    return {Code::Invalid, "num columns of t1 and t2 are not equal!"};
  }

  std::vector<int> col_ids(t1->num_columns());
  std::iota(col_ids.begin(), col_ids.end(), 0);

  return DualTableRowIndexEqualTo::Make(t1, t2, col_ids, col_ids, out_equal_to);
}

bool DualTableRowIndexEqualTo::operator()(const int64_t &record1, const int64_t &record2) const {
  return std::all_of(comparators->begin(), comparators->end(),
                     [&](const std::shared_ptr<DualArrayIndexComparator> &comp) {
                       return comp->equal_to(record1, record2);
                     });
}

int DualTableRowIndexEqualTo::compare(const int64_t &record1, const int64_t &record2) const {
  for (const auto &comp: *comparators) {
    int com = comp->compare(record1, record2);
    if (com != 0) {
      return com;
    }
  }
  return 0;
}

int DualTableRowIndexEqualTo::compare(const int32_t &table1,
                                      const int64_t &record1,
                                      const int32_t &table2,
                                      const int64_t &record2) const {
  for (const auto &comp: *comparators) {
    int com = comp->compare(table1, record1, table2, record2);
    if (com != 0) {
      return com;
    }
  }
  return 0;
}

Status ArrayIndexHash::Make(const std::shared_ptr<arrow::Array> &arr,
                            std::unique_ptr<ArrayIndexHash> *hash) {
  auto hashes_ptr = std::make_shared<std::vector<uint32_t>>(arr->length(), 0);
  std::unique_ptr<HashPartitionKernel> hash_kernel;
  RETURN_CYLON_STATUS_IF_FAILED(CreateHashPartitionKernel(arr->type(), &hash_kernel));
  // update the hashes
  RETURN_CYLON_STATUS_IF_FAILED(hash_kernel->UpdateHash(arr, *hashes_ptr));

  *hash = std::make_unique<ArrayIndexHash>(std::move(hashes_ptr));
  return Status::OK();
}

size_t ArrayIndexHash::operator()(const int64_t &record) const {
  return (*hashes_ptr)[record];
}

Status DualArrayIndexHash::Make(const std::shared_ptr<arrow::Array> &arr1,
                                const std::shared_ptr<arrow::Array> &arr2,
                                std::unique_ptr<DualArrayIndexHash> *hash) {
  std::unique_ptr<ArrayIndexHash> h1, h2;
  RETURN_CYLON_STATUS_IF_FAILED(ArrayIndexHash::Make(arr1, &h1));
  RETURN_CYLON_STATUS_IF_FAILED(ArrayIndexHash::Make(arr2, &h2));

  *hash = std::make_unique<DualArrayIndexHash>(std::move(h1), std::move(h2));
  return Status::OK();
}

size_t DualArrayIndexHash::operator()(int64_t idx) const {
  return array_hashes[util::CheckBit(idx)]->operator()(util::ClearBit(idx));
}

}  // namespace cylon
