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

#ifndef CYLON_CPP_SRC_CYLON_COMPUTE_AGGREGATE_KERNELS_HPP_
#define CYLON_CPP_SRC_CYLON_COMPUTE_AGGREGATE_KERNELS_HPP_

namespace cylon {
namespace compute {

// aggregation operation IDs
enum AggregationOpId {
  SUM,
  MIN,
  MAX,
  COUNT,
  MEAN,
  VAR
};

// kernel options
struct KernelOptions {
  virtual ~KernelOptions() = default;
};

struct EmptyKernelOptions : public KernelOptions {};

struct VarKernelOptions : public KernelOptions {
  explicit VarKernelOptions(int ddof) : ddof(ddof) {}
  VarKernelOptions() : VarKernelOptions(0) {}

  int ddof;
};

// aggregation ops
struct AggregationOp {
  explicit AggregationOp(AggregationOpId id) : id(id), options(nullptr) {}
  AggregationOp(AggregationOpId id, std::unique_ptr<KernelOptions> options) : id(id), options(std::move(options)) {}

  AggregationOpId id;
  std::unique_ptr<KernelOptions> options;
};

template<AggregationOpId ID>
struct BaseAggregationOp : public AggregationOp {
  BaseAggregationOp() : AggregationOp(ID, std::make_unique<EmptyKernelOptions>()) {}

  static inline std::unique_ptr<AggregationOp> Make() {
    return std::make_unique<BaseAggregationOp<ID>>();
  }
};

struct SumOp : public BaseAggregationOp<SUM> {};
struct MinOp : public BaseAggregationOp<MIN> {};
struct MaxOp : public BaseAggregationOp<MAX> {};
struct CountOp : public BaseAggregationOp<COUNT> {};
struct MeanOp : public BaseAggregationOp<MEAN> {};

struct VarOp : public AggregationOp {
  explicit VarOp(int ddof) : AggregationOp(VAR, std::make_unique<VarKernelOptions>(ddof)) {}

  static inline std::unique_ptr<AggregationOp> Make(int ddof = 0) {
    return std::make_unique<VarOp>(ddof);
  }
};

// kernel traits
template<AggregationOpId op, typename T>
struct KernelTraits {};

template<typename T>
struct KernelTraits<AggregationOpId::SUM, T> {
  using State = std::tuple<T>;
  using ResultT = T;
  using Options = EmptyKernelOptions;
  static constexpr const char *name() {
    return "sum_";
  }
};

template<typename T>
struct KernelTraits<AggregationOpId::MEAN, T> {
  using State = std::tuple<T, int64_t>; // <sum, count>
  using ResultT = T;
  using Options = EmptyKernelOptions;
  static constexpr const char *name() {
    return "mean_";
  }
};

template<typename T>
struct KernelTraits<AggregationOpId::VAR, T> {
  using State = std::tuple<T, T, int64_t>; // <sum of squares, sum, count>
  using ResultT = double_t;
  using Options = VarKernelOptions;
  static constexpr const char *name() {
    return "var_";
  }
};

template<typename T>
struct KernelTraits<AggregationOpId::COUNT, T> {
  using State = std::tuple<int64_t>;
  using ResultT = int64_t;
  using Options = EmptyKernelOptions;
  static constexpr const char *name() {
    return "count_";
  }
};

template<typename T>
struct KernelTraits<AggregationOpId::MIN, T> {
  using State = std::tuple<T>;
  using ResultT = T;
  using Options = EmptyKernelOptions;
  static constexpr const char *name() {
    return "min_";
  }
};

template<typename T>
struct KernelTraits<AggregationOpId::MAX, T> {
  using State = std::tuple<T>;
  using ResultT = T;
  using Options = EmptyKernelOptions;
  static constexpr const char *name() {
    return "max_";
  }
};

struct Kernel {
  virtual ~Kernel() = default;
  virtual void Setup(KernelOptions *options) = 0;
  virtual void InitializeState(void *state) = 0;
  virtual void Update(const void *value, void *state) = 0;
  virtual void Finalize(const void *state, void *result) = 0;
};

template<AggregationOpId op, typename T, typename State = typename KernelTraits<op, T>::State,
    typename ResultT = typename KernelTraits<op, T>::ResultT, typename Options = typename KernelTraits<op, T>::Options>
struct TypedKernel : Kernel {
  ~TypedKernel() override = default;
  virtual void Setup(Options *options) {}; // default implementation
  virtual void InitializeState(State *state) = 0;
  virtual void Update(const T *value, State *state) = 0;
  virtual void Finalize(const State *state, ResultT *result) = 0;

  inline void Setup(KernelOptions *options) override {
    return Setup(static_cast<Options *> (options));
  }
  inline void InitializeState(void *state) override {
    return InitializeState(static_cast<State *> (state));
  }
  inline void Update(const void *value, void *state) override {
    return Update(static_cast<const T *>(value), static_cast<State *>(state));
  };
  inline void Finalize(const void *state, void *result) override {
    return Finalize(static_cast<const State *>(state), static_cast<ResultT *>(result));
  }
};

template<typename T>
class MeanKernel : public TypedKernel<AggregationOpId::MEAN, T> {
 public:
  inline void InitializeState(std::tuple<T, int64_t> *state) override {
    *state = {0, 0};
  }
  inline void Update(const T *value, std::tuple<T, int64_t> *state) override {
    std::get<0>(*state) += *value;
    std::get<1>(*state) += 1;
  }
  inline void Finalize(const std::tuple<T, int64_t> *state, T *result) override {
    if (std::get<1>(*state) != 0) {
      *result = std::get<0>(*state) / std::get<1>(*state);
    }
  }
};

template<typename T>
class VarianceKernel : public TypedKernel<AggregationOpId::VAR, T> {
 public:
  void Setup(VarKernelOptions *options) override {
    ddof = options->ddof;
  }

  inline void InitializeState(std::tuple<T, T, int64_t> *state) override {
    *state = {0, 0, 0};
  }
  inline void Update(const T *value, std::tuple<T, T, int64_t> *state) override {
    std::get<0>(*state) += (*value) * (*value);
    std::get<1>(*state) += *value;
    std::get<2>(*state) += 1;
  }
  inline void Finalize(const std::tuple<T, T, int64_t> *state, double *result) override {
    if (std::get<2>(*state) == 1) {
      *result = 0;
    } else if (std::get<2>(*state) != 0) {
      double div = std::get<2>(*state) - ddof;
      double mean = static_cast<double>(std::get<1>(*state)) / div;
      *result = static_cast<double>(std::get<0>(*state)) / div - mean * mean;
    }
  };

 private:
  int ddof;
};

template<typename T>
struct SumKernel : public TypedKernel<AggregationOpId::SUM, T> {
  inline void InitializeState(std::tuple<T> *state) override {
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
struct CountKernel : public TypedKernel<AggregationOpId::COUNT, T> {
  inline void InitializeState(std::tuple<int64_t> *state) override {
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
struct MinKernel : public TypedKernel<AggregationOpId::MIN, T> {
  inline void InitializeState(std::tuple<T> *state) override {
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
struct MaxKernel : public TypedKernel<AggregationOpId::MAX, T> {
  inline void InitializeState(std::tuple<T> *state) override {
    *state = {std::numeric_limits<T>::min()};
  }
  inline void Update(const T *value, std::tuple<T> *state) override {
    std::get<0>(*state) = std::max(*value, std::get<0>(*state));
  }
  inline void Finalize(const std::tuple<T> *state, T *result) override {
    *result = std::get<0>(*state);
  }
};

template<AggregationOpId op, typename T>
std::unique_ptr<Kernel> CreateAggregateKernel() {
  switch (op) {
    case SUM: return std::make_unique<SumKernel<T>>();
    case MIN:return std::make_unique<MinKernel<T>>();
    case MAX:return std::make_unique<MaxKernel<T>>();
    case COUNT:return std::make_unique<CountKernel<T>>();
    case MEAN:return std::make_unique<MeanKernel<T>>();
    case VAR:return std::make_unique<VarianceKernel<T>>();
  }
}

}
}

#endif //CYLON_CPP_SRC_CYLON_COMPUTE_AGGREGATE_KERNELS_HPP_
