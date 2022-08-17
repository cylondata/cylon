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

#ifndef CYLON_CPP_SRC_CYLON_UTIL_ARROW_RAND_HPP_
#define CYLON_CPP_SRC_CYLON_UTIL_ARROW_RAND_HPP_

#include <arrow/api.h>
#include <random>

namespace cylon {

namespace {
template<typename V>
typename std::enable_if_t<std::is_integral<V>::value, void> GenerateTypedData(
    V *data, size_t n, V &min_, V &max_, uint32_t seed = 0) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<V> dist(min_, max_);

  // A static cast is required due to the int16 -> int8 handling.
  std::generate(data, data + n, [&] { return static_cast<V>(dist(rng)); });
}

template<typename V>
typename std::enable_if_t<std::is_floating_point<V>::value, void> GenerateTypedData(
    V *data, size_t n, V &min_, V &max_, uint32_t seed = 0) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<V> dist(min_, max_);

  // A static cast is required due to the int16 -> int8 handling.
  std::generate(data, data + n, [&] { return static_cast<V>(dist(rng)); });
}

void GenerateBitmap(uint8_t *buffer,
                    size_t n,
                    double false_prob,
                    int64_t *false_count,
                    uint32_t seed = 0) {
  int64_t count = 0;
  std::mt19937 rng(seed);
  std::bernoulli_distribution dist(1.0 - false_prob);

  for (size_t i = 0; i < n; i++) {
    if (dist(rng)) {
      arrow::bit_util::SetBit(buffer, i);
    } else {
      count++;
    }
  }

  if (false_count != nullptr) *false_count = count;
}
} // namespace

struct RandomArrayGenerator {
 public:
  explicit RandomArrayGenerator(uint32_t seed = 0,
                                arrow::MemoryPool *pool = arrow::default_memory_pool())
      : seed(seed), pool(pool) {}

  std::shared_ptr<arrow::Array> Boolean(int64_t size,
                                        double true_probability,
                                        double null_probability = 0.0) {
    std::shared_ptr<arrow::Buffer> validity = nullptr, data;
    int64_t null_count = 0;
    if (null_probability > 0.0) {
      validity = *arrow::AllocateEmptyBitmap(size, pool);
      GenerateBitmap(validity->mutable_data(), size, null_probability, &null_count, seed++);
    }

    data = *arrow::AllocateEmptyBitmap(size, pool);
    if (true_probability > 0.0) {
      GenerateBitmap(validity->mutable_data(), size, 1 - true_probability, nullptr, seed++);
    }

    auto array_data = arrow::ArrayData::Make(arrow::boolean(),
                                             size,
                                             {std::move(validity), std::move(data)},
                                             null_count);
    return std::make_shared<arrow::BooleanArray>(array_data);
  }

  template<typename ArrowType>
  std::shared_ptr<arrow::Array> Numeric(int64_t size,
                                        typename ArrowType::c_type min,
                                        typename ArrowType::c_type max,
                                        double null_probability = 0.0) {
    using CType = typename ArrowType::c_type;

    std::shared_ptr<arrow::Buffer> validity = nullptr, data;
    int64_t null_count = 0;
    if (null_probability > 0.0) {
      validity = *arrow::AllocateEmptyBitmap(size, pool);
      GenerateBitmap(validity->mutable_data(), size, null_probability, &null_count, seed++);
    }

    data = *arrow::AllocateBuffer(size * sizeof(CType), pool);
    auto *mut_data = reinterpret_cast<CType *>(data->mutable_data());
    GenerateTypedData<CType>(mut_data, size, min, max, seed++);

    auto array_data = arrow::ArrayData::Make(arrow::TypeTraits<ArrowType>::type_singleton(),
                                             size,
                                             {std::move(validity), std::move(data)},
                                             null_count);
    return std::make_shared<arrow::NumericArray<ArrowType>>(std::move(array_data));
  }

 private:
  uint32_t seed;
  arrow::MemoryPool *pool;
};

}

#endif //CYLON_CPP_SRC_CYLON_UTIL_ARROW_RAND_HPP_
