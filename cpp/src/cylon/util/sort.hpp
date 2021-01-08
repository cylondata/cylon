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
#ifndef CYLON_SRC_CYLON_UTIL_SORT_H_
#define CYLON_SRC_CYLON_UTIL_SORT_H_

#include <cmath>
#include <arrow/api.h>

namespace cylon {
namespace util {

class SwapFunction {
 public:
  explicit SwapFunction() = default;;
  virtual void swap(int64_t i, int64_t j) = 0;
};

template<typename TYPE>
class NumericSwapFunction : public SwapFunction {
 public:
  using T = typename TYPE::c_type;

  explicit NumericSwapFunction(std::shared_ptr<arrow::Array> values) {
    auto array = std::static_pointer_cast<arrow::NumericArray<TYPE>>(values);
    std::shared_ptr<arrow::ArrayData> data = array->data();
    // get the first buffer as a mutable buffer
    left_data = data->GetMutableValues<T>(1);
  }

  void swap(int64_t i, int64_t j) override {
    std::swap(left_data[i], left_data[j]);
  }
 private:
  T *left_data;
};

using UInt8SwapFunction = NumericSwapFunction<arrow::UInt8Type>;
using UInt16SwapFunction = NumericSwapFunction<arrow::UInt16Type>;
using UInt32SwapFunction = NumericSwapFunction<arrow::UInt32Type>;
using UInt64SwapFunction = NumericSwapFunction<arrow::UInt64Type>;
using Int8SwapFunction = NumericSwapFunction<arrow::Int8Type>;
using Int16SwapFunction = NumericSwapFunction<arrow::Int16Type>;
using Int32SwapFunction = NumericSwapFunction<arrow::Int32Type>;
using Int64SwapFunction = NumericSwapFunction<arrow::Int64Type>;
using HalfFloatSwapFunction = NumericSwapFunction<arrow::HalfFloatType>;
using FloatSwapFunction = NumericSwapFunction<arrow::FloatType>;
using DoubleSwapFunction = NumericSwapFunction<arrow::DoubleType>;

template <typename T, typename  T2>
void insertion_sort(T *t, T2 *t2, int start, int end) {
  for (int i = start + 1; i <= end; i++) {
    T value = t[i];
    T value2 = t2[i];
    int j = i;

    while (j > start && t[j - 1] > value) {
      t[j] = t[j - 1];
      t2[j] = t2[j - 1];
      j--;
    }
    t[j] = value;
    t2[j] = value2;
  }
}

template <typename T, typename  T2>
void heapify(T *arr, int n, int i, T2 *t2) {
  int largest = i;
  int l = 2 * i + 1;
  int r = 2 * i + 2;
  if (l < n && arr[l] > arr[largest]) {
    largest = l;
  }
  if (r < n && arr[r] > arr[largest]) {
    largest = r;
  }
  if (largest != i) {
    std::swap(arr[i], arr[largest]);
    std::swap(t2[i], t2[largest]);
    heapify(arr, n, largest, t2);
  }
}

template <typename T, typename  T2>
void heap_sort(T *arr, int length, T *t2) {
  for (int i = length / 2 - 1; i >= 0; i--)
    heapify(arr, length, i);

  for (int i=length-1; i>0; i--) {
    std::swap(&arr[0], &arr[i]);
    heapify(arr, i, 0, t2);
  }
}


template <typename T, typename  T2>
int64_t partition(T *t, T2 *t2, int start, int end) {
  T pivot = t[end];
  int i = start - 1;
  for (int j = start; j < end; j++) {
    if (t[j] <= pivot) {
      i++;
      std::swap(t2[i], t2[j]);
      std::swap(t[i], t[j]);
    }
  }
  std::swap(t[i + 1], t[end]);
  std::swap(t2[i + 1], t2[end]);
  return i + 1;
}

template <typename T, typename  T2>
void introsort_impl(T *t, T2 *t2, int64_t start, int64_t end, int maxdepth) {
  if (end - start < 32) {
    insertion_sort(t, start, end);
  } else if (maxdepth == 0) {
    heap_sort(t, start, end);
  } else {
    int64_t p = partition(t, t2, start, end);
    introsort_impl(t, start, p - 1, maxdepth - 1);
    introsort_impl(t, p + 1, end, maxdepth - 1);
  }
}

template <typename T, typename  T2>
void introsort(T *t, T2 *t2, int64_t length) {
  int64_t depth = std::log(length) * 2;
  introsort_impl(t, t2, 0, length - 1, depth);
}

template <typename T, typename  T2>
void quicksort_imp(T *arr, int low, int high, T2 *t2) {
  if (low < high) {
    int p = partition(arr, t2, low, high);
    quicksort_imp(arr, low, p - 1, t2);
    quicksort_imp(arr, p + 1, high, t2);
  }
}

template <typename T, typename  T2>
void quicksort(T *arr, int low, int high, T2 *t2) {
  quicksort_imp(arr, low, high - 1, t2);
}

}  // namespace util
}  // namespace cylon


#endif //CYLON_SRC_CYLON_UTIL_SORT_H_
