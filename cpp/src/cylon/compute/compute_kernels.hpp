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

#ifndef CYLON_CPP_SRC_CYLON_COMPUTE_COMPUTE_KERNELS_HPP_
#define CYLON_CPP_SRC_CYLON_COMPUTE_COMPUTE_KERNELS_HPP_

namespace cylon {
namespace compute {

enum AggregationOp {
  SUM,
  MIN,
  MAX,
  COUNT,
  MEAN,
};

template<AggregationOp op, typename T>
struct KernelTraits {};

template<typename T>
struct KernelTraits<AggregationOp::SUM, T> {
  using State = std::tuple<T>;
  using ResultT = T;
  static constexpr const char *name() {
    return "sum_";
  }
};

template<typename T>
struct KernelTraits<AggregationOp::MEAN, T> {
  using State = std::tuple<T, int64_t>;
  using ResultT = T;
  static constexpr const char *name() {
    return "mean_";
  }
};

template<typename T>
struct KernelTraits<AggregationOp::COUNT, T> {
  using State = std::tuple<int64_t>;
  using ResultT = int64_t;
  static constexpr const char *name() {
    return "count_";
  }
};

template<typename T>
struct KernelTraits<AggregationOp::MIN, T> {
  using State = std::tuple<T>;
  using ResultT = T;
  static constexpr const char *name() {
    return "min_";
  }
};

template<typename T>
struct KernelTraits<AggregationOp::MAX, T> {
  using State = std::tuple<T>;
  using ResultT = T;
  static constexpr const char *name() {
    return "max_";
  }
};

struct Kernel {
  virtual void Init(void *state) = 0;
  virtual void Update(const void *value, void *state) = 0;
  virtual void Finalize(const void *state, void *result) = 0;
};

template<AggregationOp op, typename T, typename State = typename KernelTraits<op, T>::State,
    typename ResultT = typename KernelTraits<op, T>::ResultT>
struct TypedKernel : Kernel {
  virtual void Init(State *state) = 0;
  virtual void Update(const T *value, State *state) = 0;
  virtual void Finalize(const State *state, ResultT *result) = 0;

  inline void Init(void *state) override {
    return Init(static_cast<State *> (state));
  }
  inline void Update(const void *value, void *state) override {
    return Update(static_cast<const T *>(value), static_cast<State *>(state));
  };
  inline void Finalize(const void *state, void *result) override {
    return Finalize(static_cast<const State *>(state), static_cast<ResultT *>(result));
  }
};

template<typename T>
class MeanKernel : public TypedKernel<AggregationOp::MEAN, T> {
 public:
  inline void Init(std::tuple<T, int64_t> *state) override {
    *state = {0, 0};
  }
  inline void Update(const T *value, std::tuple<T, int64_t> *state) override {
    std::get<0>(*state) += *value;
    std::get<1>(*state) += 1;
  }
  inline void Finalize(const std::tuple<T, int64_t> *state, T *result) override {
    *result = std::get<0>(*state) / std::get<1>(*state);
  }
};

template<typename T>
struct SumKernel : public TypedKernel<AggregationOp::SUM, T> {
  inline void Init(std::tuple<T> *state) override {
    *state = {0};
  }
  inline void Update(const T *value, std::tuple<T> *state) override {
    std::get<0>(*state) += *value;
  }
  inline void Finalize(const std::tuple<T> *state, T *result) override {
    *result = std::get<0>(*state);
  }
};

template<typename T>
struct CountKernel : public TypedKernel<AggregationOp::COUNT, T> {
  inline void Init(std::tuple<int64_t> *state) override {
    *state = {0};
  }
  inline void Update(const T *value, std::tuple<int64_t> *state) override {
    std::get<0>(*state) += 1;
  }
  inline void Finalize(const std::tuple<int64_t> *state, long *result) override {
    *result = std::get<0>(*state);
  }
};

template<typename T>
struct MinKernel : public TypedKernel<AggregationOp::MIN, T> {
  inline void Init(std::tuple<T> *state) override {
    *state = {std::numeric_limits<T>::max()};
  }
  inline void Update(const T *value, std::tuple<T> *state) override {
    std::get<0>(*state) = std::min(*value, std::get<0>(*state));
  }
  inline void Finalize(const std::tuple<T> *state, T *result) override {
    *result = std::get<0>(*state);
  }
};

template<typename T>
struct MaxKernel : public TypedKernel<AggregationOp::MAX, T> {
  inline void Init(std::tuple<T> *state) override {
    *state = {std::numeric_limits<T>::min()};
  }
  inline void Update(const T *value, std::tuple<T> *state) override {
    std::get<0>(*state) = std::max(*value, std::get<0>(*state));
  }
  inline void Finalize(const std::tuple<T> *state, T *result) override {
    *result = std::get<0>(*state);
  }
};

template<AggregationOp op, typename T>
std::unique_ptr<Kernel> CreateAggregateKernel() {
  switch (op) {
    case SUM: return std::make_unique<SumKernel<T>>();
    case MIN:return std::make_unique<MinKernel<T>>();
    case MAX:return std::make_unique<MaxKernel<T>>();
    case COUNT:return std::make_unique<CountKernel<T>>();
    case MEAN:return std::make_unique<MeanKernel<T>>();
  }
}

}
}

#endif //CYLON_CPP_SRC_CYLON_COMPUTE_COMPUTE_KERNELS_HPP_
