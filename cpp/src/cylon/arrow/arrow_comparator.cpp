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

#include <memory>
#include <vector>

namespace cylon {

template<typename ARROW_TYPE>
class NumericArrowComparator : public ArrowComparator {
  int compare(std::shared_ptr<arrow::Array> array1,
              int64_t index1,
              std::shared_ptr<arrow::Array> array2,
              int64_t index2) override {
    auto reader1 = std::static_pointer_cast<arrow::NumericArray<ARROW_TYPE>>(array1);
    auto reader2 = std::static_pointer_cast<arrow::NumericArray<ARROW_TYPE>>(array2);

    auto value1 = reader1->Value(index1);
    auto value2 = reader2->Value(index2);

    if (value1 < value2) {
      return -1;
    } else if (value1 > value2) {
      return 1;
    }
    return 0;
  }
};

class BinaryArrowComparator : public ArrowComparator {
  int compare(std::shared_ptr<arrow::Array> array1,
              int64_t index1,
              std::shared_ptr<arrow::Array> array2,
              int64_t index2) override {
    auto reader1 = std::static_pointer_cast<arrow::BinaryArray>(array1);
    auto reader2 = std::static_pointer_cast<arrow::BinaryArray>(array2);

    auto value1 = reader1->GetString(index1);
    auto value2 = reader2->GetString(index2);

    return value1.compare(value2);
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
    case arrow::Type::HALF_FLOAT:
      return std::make_shared<NumericArrowComparator<arrow::HalfFloatType>>();
    case arrow::Type::FLOAT:return std::make_shared<NumericArrowComparator<arrow::FloatType>>();
    case arrow::Type::DOUBLE:return std::make_shared<NumericArrowComparator<arrow::DoubleType>>();
    case arrow::Type::STRING:return std::make_shared<BinaryArrowComparator>();
    case arrow::Type::BINARY:return std::make_shared<BinaryArrowComparator>();
    case arrow::Type::FIXED_SIZE_BINARY:break;
    case arrow::Type::DATE32:break;
    case arrow::Type::DATE64:break;
    case arrow::Type::TIMESTAMP:break;
    case arrow::Type::TIME32:break;
    case arrow::Type::TIME64:break;
    case arrow::Type::INTERVAL:break;
    case arrow::Type::DECIMAL:break;
    case arrow::Type::LIST:break;
    case arrow::Type::STRUCT:break;
    case arrow::Type::UNION:break;
    case arrow::Type::DICTIONARY:break;
    case arrow::Type::MAP:break;
    case arrow::Type::EXTENSION:break;
    case arrow::Type::FIXED_SIZE_LIST:break;
    case arrow::Type::DURATION:break;
    case arrow::Type::LARGE_STRING:break;
    case arrow::Type::LARGE_BINARY:break;
    case arrow::Type::LARGE_LIST:break;
  }
  return nullptr;
}

TableRowComparator::TableRowComparator(const std::vector<std::shared_ptr<arrow::Field>> &fields) {
  for (const auto &field : fields) {
    this->comparators.push_back(GetComparator(field->type()));
  }
}

int TableRowComparator::compare(const std::shared_ptr<arrow::Table> &table1,
                                int64_t index1,
                                const std::shared_ptr<arrow::Table> &table2,
                                int64_t index2) {
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
}  // namespace cylon
