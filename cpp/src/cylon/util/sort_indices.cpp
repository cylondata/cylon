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
#include <arrow/compare.h>
#include <arrow/visitor_inline.h>

#include <numeric>
#include <chrono>

#include "arrow_utils.hpp"

/**
 * This class is direct copy from arrow to measure the difference between stable sort and sort
 */

namespace cylon {
namespace util {

class SortToIndicesKernel : public arrow::compute::UnaryKernel {
 protected:
  std::shared_ptr<arrow::DataType> type_;

 public:
  /// \brief UnaryKernel interface
  ///
  /// delegates to subclasses via SortToIndices()
  arrow::Status Call(arrow::compute::FunctionContext *ctx, const arrow::compute::Datum &values,
                     arrow::compute::Datum *offsets) override = 0;

  /// \brief output type of this kernel
  std::shared_ptr<arrow::DataType> out_type() const override {
    return arrow::uint64();
  }

  /// \brief single-array implementation
  virtual arrow::Status
  SortToIndices(arrow::compute::FunctionContext *ctx, const std::shared_ptr<arrow::Array> &values,
                std::shared_ptr<arrow::Array> *offsets) = 0;

  /// \brief factory for SortToIndicesKernel
  ///
  /// \param[in] value_type constructed SortToIndicesKernel will support sorting
  ///            values of this type
  /// \param[out] out created kernel
  static arrow::Status Make(const std::shared_ptr<arrow::DataType> &value_type,
                            std::shared_ptr<SortToIndicesKernel> *out);
};
template<typename ArrayType>
bool CompareValues(const ArrayType &array, uint64_t lhs, uint64_t rhs) {
  return array.Value(lhs) < array.Value(rhs);
}

template<typename ArrayType>
bool CompareViews(const ArrayType &array, uint64_t lhs, uint64_t rhs) {
  return array.GetView(lhs) < array.GetView(rhs);
}

template<typename ArrowType, typename Comparator>
class CompareSorter {
  using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

 public:
  explicit CompareSorter(Comparator compare) : compare_(compare) {}

  void Sort(int64_t *indices_begin, int64_t *indices_end, const ArrayType &values) {
    std::iota(indices_begin, indices_end, 0);

    auto nulls_begin = indices_end;
    if (values.null_count()) {
      nulls_begin =
          std::stable_partition(indices_begin, indices_end,
                                [&values](uint64_t ind) { return !values.IsNull(ind); });
    }
    auto start3 = std::chrono::high_resolution_clock::now();
    std::sort(indices_begin, nulls_begin,
              [&values, this](uint64_t left, uint64_t right) {
                return compare_(values, left, right);
              });
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
    LOG(INFO) << "Arrow sorting time: " + std::to_string(duration4.count());
  }

 private:
  Comparator compare_;
};

template<typename ArrowType>
class CountSorter {
  using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
  using c_type = typename ArrowType::c_type;

 public:
  CountSorter() = default;

  explicit CountSorter(c_type min, c_type max) { SetMinMax(min, max); }

  // Assume: max >= min && (max - min) < 4Gi
  void SetMinMax(c_type min, c_type max) {
    min_ = min;
    value_range_ = static_cast<uint32_t>(max - min) + 1;
  }

  void Sort(int64_t *indices_begin, int64_t *indices_end, const ArrayType &values) {
    // 32bit counter performs much better than 64bit one
    if (values.length() < (1LL << 32)) {
      SortInternal<uint32_t>(indices_begin, indices_end, values);
    } else {
      SortInternal<uint64_t>(indices_begin, indices_end, values);
    }
  }

 private:
  c_type min_{0};
  uint32_t value_range_{0};

  template<typename CounterType>
  void SortInternal(int64_t *indices_begin, int64_t *indices_end,
                    const ArrayType &values) {
    const uint32_t value_range = value_range_;

    // first slot reserved for prefix sum, last slot for null value
    std::vector<CounterType> counts(1 + value_range + 1);

    struct UpdateCounts {
      arrow::Status VisitNull() {
        ++counts[value_range];
        return arrow::Status::OK();
      }

      arrow::Status VisitValue(c_type v) {
        ++counts[v - min];
        return arrow::Status::OK();
      }

      CounterType *counts;
      const uint32_t value_range;
      c_type min;
    };
    {
      UpdateCounts update_counts{&counts[1], value_range, min_};
      ARROW_CHECK_OK(arrow::ArrayDataVisitor<ArrowType>().Visit(*values.data(), &update_counts));
    }

    for (uint32_t i = 1; i <= value_range; ++i) {
      counts[i] += counts[i - 1];
    }

    struct OutputIndices {
      arrow::Status VisitNull() {
        out_indices[counts[value_range]++] = index++;
        return arrow::Status::OK();
      }

      arrow::Status VisitValue(c_type v) {
        out_indices[counts[v - min]++] = index++;
        return arrow::Status::OK();
      }

      CounterType *counts;
      const uint32_t value_range;
      c_type min;
      int64_t *out_indices;
      int64_t index;
    };
    {
      OutputIndices output_indices{&counts[0], value_range, min_, indices_begin, 0};
      ARROW_CHECK_OK(
          arrow::ArrayDataVisitor<ArrowType>().Visit(*values.data(), &output_indices));
    }
  }
};

// Sort integers with counting sort or comparison based sorting algorithm
// - Use O(n) counting sort if values are in a small range
// - Use O(nlogn) std::stable_sort otherwise
template<typename ArrowType, typename Comparator>
class CountOrCompareSorter {
  using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
  using c_type = typename ArrowType::c_type;

 public:
  explicit CountOrCompareSorter(Comparator compare) : compare_sorter_(compare) {}

  void Sort(int64_t *indices_begin, int64_t *indices_end, const ArrayType &values) {
    if (values.length() >= countsort_min_len_ && values.length() > values.null_count()) {
      struct MinMaxScanner {
        arrow::Status VisitNull() { return arrow::Status::OK(); }

        arrow::Status VisitValue(c_type v) {
          min = std::min(min, v);
          max = std::max(max, v);
          return arrow::Status::OK();
        }

        c_type min{std::numeric_limits<c_type>::max()};
        c_type max{std::numeric_limits<c_type>::min()};
      };

      MinMaxScanner minmax_scanner;
      ARROW_CHECK_OK(
          arrow::ArrayDataVisitor<ArrowType>().Visit(*values.data(), &minmax_scanner));

      // For signed int32/64, (max - min) may overflow and trigger UBSAN.
      // Cast to largest unsigned type(uint64_t) before substraction.
      const uint64_t min = static_cast<uint64_t>(minmax_scanner.min);
      const uint64_t max = static_cast<uint64_t>(minmax_scanner.max);
      if ((max - min) <= countsort_max_range_) {
        count_sorter_.SetMinMax(minmax_scanner.min, minmax_scanner.max);
        count_sorter_.Sort(indices_begin, indices_end, values);
        return;
      }
    }

    compare_sorter_.Sort(indices_begin, indices_end, values);
  }

 private:
  CompareSorter<ArrowType, Comparator> compare_sorter_;
  CountSorter<ArrowType> count_sorter_;

  // Cross point to prefer counting sort than stl::stable_sort(merge sort)
  // - array to be sorted is longer than "count_min_len_"
  // - value range (max-min) is within "count_max_range_"
  //
  // The optimal setting depends heavily on running CPU. Below setting is
  // conservative to adapt to various hardware and keep code simple.
  // It's possible to decrease array-len and/or increase value-range to cover
  // more cases, or setup a table for best array-len/value-range combinations.
  // See https://issues.apache.org/jira/browse/ARROW-1571 for detailed analysis.
  static const uint32_t countsort_min_len_ = 1024;
  static const uint32_t countsort_max_range_ = 4096;
};

template<typename ArrowType, typename Sorter>
class SortToIndicesKernelImpl : public SortToIndicesKernel {
  using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;

 public:
  explicit SortToIndicesKernelImpl(Sorter sorter) : sorter_(sorter) {}

  arrow::Status SortToIndices(arrow::compute::FunctionContext *ctx,
                              const std::shared_ptr<arrow::Array> &values,
                              std::shared_ptr<arrow::Array> *offsets) {
    return SortToIndicesImpl(ctx, std::static_pointer_cast<ArrayType>(values), offsets);
  }

  arrow::Status Call(arrow::compute::FunctionContext *ctx,
                     const arrow::compute::Datum &values, arrow::compute::Datum *offsets) {
    if (!values.is_array()) {
      return arrow::Status::Invalid("SortToIndicesKernel expects array values");
    }
    auto values_array = values.make_array();
    std::shared_ptr<arrow::Array> offsets_array;
    RETURN_NOT_OK(this->SortToIndices(ctx, values_array, &offsets_array));
    *offsets = offsets_array;
    return arrow::Status::OK();
  }

  std::shared_ptr<arrow::DataType> out_type() const { return type_; }

 private:
  Sorter sorter_;

  arrow::Status SortToIndicesImpl(arrow::compute::FunctionContext *ctx,
                                  const std::shared_ptr<ArrayType> &values,
                                  std::shared_ptr<arrow::Array> *offsets) {
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = values->length() * sizeof(uint64_t);
    RETURN_NOT_OK(AllocateBuffer(arrow::default_memory_pool(), buf_size, &indices_buf));

    int64_t *indices_begin = reinterpret_cast<int64_t *>(indices_buf->mutable_data());
    int64_t *indices_end = indices_begin + values->length();

    sorter_.Sort(indices_begin, indices_end, *values.get());
    *offsets = std::make_shared<arrow::UInt64Array>(values->length(), indices_buf);
    return arrow::Status::OK();
  }
};

template<typename ArrowType, typename Comparator,
    typename Sorter = CompareSorter<ArrowType, Comparator>>
SortToIndicesKernelImpl<ArrowType, Sorter> *MakeCompareKernel(Comparator comparator) {
  return new SortToIndicesKernelImpl<ArrowType, Sorter>(Sorter(comparator));
}

template<typename ArrowType, typename Sorter = CountSorter<ArrowType>>
SortToIndicesKernelImpl<ArrowType, Sorter> *MakeCountKernel(int min, int max) {
  return new SortToIndicesKernelImpl<ArrowType, Sorter>(Sorter(min, max));
}

template<typename ArrowType, typename Comparator,
    typename Sorter = CountOrCompareSorter<ArrowType, Comparator>>
SortToIndicesKernelImpl<ArrowType, Sorter> *MakeCountOrCompareKernel(
    Comparator comparator) {
  return new SortToIndicesKernelImpl<ArrowType, Sorter>(Sorter(comparator));
}

arrow::Status SortToIndicesKernel::Make(const std::shared_ptr<arrow::DataType> &value_type,
                                        std::shared_ptr<SortToIndicesKernel> *out) {
  SortToIndicesKernel *kernel;
  switch (value_type->id()) {
    case arrow::Type::UINT8:kernel = MakeCountKernel<arrow::UInt8Type>(0, 255);
      break;
    case arrow::Type::INT8:kernel = MakeCountKernel<arrow::Int8Type>(-128, 127);
      break;
    case arrow::Type::UINT16:kernel = MakeCountOrCompareKernel<arrow::UInt16Type>(
        CompareValues<arrow::UInt16Array>);
      break;
    case arrow::Type::INT16:kernel = MakeCountOrCompareKernel<arrow::Int16Type>(
        CompareValues<arrow::Int16Array>);
      break;
    case arrow::Type::UINT32:kernel = MakeCountOrCompareKernel<arrow::UInt32Type>(
        CompareValues<arrow::UInt32Array>);
      break;
    case arrow::Type::INT32:kernel = MakeCountOrCompareKernel<arrow::Int32Type>(
        CompareValues<arrow::Int32Array>);
      break;
    case arrow::Type::UINT64:kernel = MakeCountOrCompareKernel<arrow::UInt64Type>(
        CompareValues<arrow::UInt64Array>);
      break;
    case arrow::Type::INT64:kernel = MakeCountOrCompareKernel<arrow::Int64Type>(
        CompareValues<arrow::Int64Array>);
      break;
    case arrow::Type::FLOAT:kernel = MakeCompareKernel<arrow::FloatType>(
        CompareValues<arrow::FloatArray>);
      break;
    case arrow::Type::DOUBLE:kernel = MakeCompareKernel<arrow::DoubleType>(
        CompareValues<arrow::DoubleArray>);
      break;
    case arrow::Type::BINARY:kernel = MakeCompareKernel<arrow::BinaryType>(
        CompareViews<arrow::BinaryArray>);
      break;
    case arrow::Type::STRING:kernel = MakeCompareKernel<arrow::StringType>(
        CompareViews<arrow::StringArray>);
      break;
    default:return arrow::Status::NotImplemented("Sorting of ", *value_type, " arrays");
  }
  out->reset(kernel);
  return arrow::Status::OK();
}

arrow::Status SortToIndices(arrow::compute::FunctionContext *ctx,
                            const arrow::compute::Datum &values,
                            arrow::compute::Datum *offsets) {
  std::shared_ptr<SortToIndicesKernel> kernel;
  RETURN_NOT_OK(SortToIndicesKernel::Make(values.type(), &kernel));
  return kernel->Call(ctx, values, offsets);
}

arrow::Status SortToIndices(arrow::compute::FunctionContext *ctx, const arrow::Array &values,
                            std::shared_ptr<arrow::Array> *offsets) {
  arrow::compute::Datum offsets_datum;
  RETURN_NOT_OK(SortToIndices(ctx, arrow::compute::Datum(values.data()), &offsets_datum));
  *offsets = offsets_datum.make_array();
  return arrow::Status::OK();
}

}  // namespace util
}  // namespace cylon
