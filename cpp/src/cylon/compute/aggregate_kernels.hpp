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

#include <cmath>

namespace cylon {
namespace compute {

/**
 * Cylon Aggregations
 *
 * Aggregations are of 3 components.
 *  1. Aggregation Operation ID (AggregationOpId)
 *  2. Aggregation Operation (AggregationOpId) - Struct with AggregationOpId + KernelOptions
 *  3. Aggregation Kernel (AggregationKernel) - Stateless AggregationOp execution
 *
 *  For every kernel, there is a traits struct (KernelTrait<AggregationOpId, T>) that would be used to cast the pointers
 *  passed on to the AggregationKernel.
 */

/**
 * aggregation operation IDs
 */
enum AggregationOpId {
  SUM,
  MIN,
  MAX,
  COUNT,
  MEAN,
  VAR
};

/**
 * 1. Kernel options holder
 */
struct KernelOptions {
  virtual ~KernelOptions() = default;
};

/**
 * special kernel options holder for default options. i.e. no special options.
 */
struct DefaultKernelOptions : public KernelOptions {};

/**
 * Variance kernel options
 */
struct VarKernelOptions : public KernelOptions {
  /**
   * @param ddof delta degree of freedom
   */
  explicit VarKernelOptions(int ddof) : ddof(ddof) {}
  VarKernelOptions() : VarKernelOptions(0) {}

  int ddof;
};

// -----------------------------------------------------------------------------

/**
 * 2. Aggregation operation
 */
struct AggregationOp {
  /**
   * @param id AggregationOpId
   * @param options unique_ptr of options
   */
  AggregationOp(AggregationOpId id, std::unique_ptr<KernelOptions> options) : id(id), options(std::move(options)) {}

  /**
   * Constructor with uninitialized options. This would be explicitly handled in impl
   * @param id
   */
  explicit AggregationOp(AggregationOpId id) : id(id), options(nullptr) {}

  AggregationOpId id;
  std::unique_ptr<KernelOptions> options;
};

/**
 * Base aggregation operation with DefaultKernelOptions
 * @tparam ID AggregationOpId
 */
template<AggregationOpId ID>
struct BaseAggregationOp : public AggregationOp {
  BaseAggregationOp() : AggregationOp(ID, std::make_unique<DefaultKernelOptions>()) {}

  static inline std::unique_ptr<AggregationOp> Make() {
    return std::make_unique<BaseAggregationOp<ID>>();
  }
};

/**
 * Sum, min, max, count, mean ops
 */
struct SumOp : public BaseAggregationOp<SUM> {};
struct MinOp : public BaseAggregationOp<MIN> {};
struct MaxOp : public BaseAggregationOp<MAX> {};
struct CountOp : public BaseAggregationOp<COUNT> {};
struct MeanOp : public BaseAggregationOp<MEAN> {};

/**
 * Var op
 */
struct VarOp : public AggregationOp {
  /**
   * @param ddof delta degree of freedom
   */
  explicit VarOp(int ddof) : AggregationOp(VAR, std::make_unique<VarKernelOptions>(ddof)) {}

  static inline std::unique_ptr<AggregationOp> Make(int ddof = 0) {
    return std::make_unique<VarOp>(ddof);
  }
};

// -----------------------------------------------------------------------------

/**
 * Kernel traits information - Helps casting the pointers passed on to the AggregationKernel
 *
 * There are 4 main traits defined statically.
 * 1. State - Type of object that would hold running state/ partial aggregation value
 * 2. ResulT - Result type of the aggregation
 * 3. Options - Options type of the aggregation
 * 4. name - (const char*) a string that would be prepended to the aggregation result column name
 */

/**
 * @tparam op AggregationOpId
 * @tparam T C type of arrow::Array value
 */
template<AggregationOpId op, typename T>
struct KernelTraits {};

template<typename T>
struct KernelTraits<AggregationOpId::SUM, T> {
  using State = std::tuple<T>; // <running sum>
  using ResultT = T;
  using Options = DefaultKernelOptions;
  static constexpr const char *name() { return "sum_"; }

};

template<typename T>
struct KernelTraits<AggregationOpId::MEAN, T> {
  using State = std::tuple<T, int64_t>; // <running sum, running count>
  using ResultT = T;
  using Options = DefaultKernelOptions;
  static constexpr const char *name() { return "mean_"; }
};

template<typename T>
struct KernelTraits<AggregationOpId::VAR, T> {
  using State = std::tuple<T, T, int64_t>; // <running sum of squares, running sum, running count>
  using ResultT = double_t;
  using Options = VarKernelOptions;
  static constexpr const char *name() { return "var_"; }
};

template<typename T>
struct KernelTraits<AggregationOpId::COUNT, T> {
  using State = std::tuple<int64_t>;
  using ResultT = int64_t;
  using Options = DefaultKernelOptions;
  static constexpr const char *name() { return "count_"; }
};

template<typename T>
struct KernelTraits<AggregationOpId::MIN, T> {
  using State = std::tuple<T>;
  using ResultT = T;
  using Options = DefaultKernelOptions;
  static constexpr const char *name() { return "min_"; }
};

template<typename T>
struct KernelTraits<AggregationOpId::MAX, T> {
  using State = std::tuple<T>;
  using ResultT = T;
  using Options = DefaultKernelOptions;
  static constexpr const char *name() { return "max_"; }
};

// -----------------------------------------------------------------------------

/**
 * 3. Aggregation Kernel
 */

struct AggregationKernel {
  // destructor defined virtual so that the derived AggregationKernels would be deleted properly
  // https://www.geeksforgeeks.org/virtual-destructor/
  virtual ~AggregationKernel() = default;

  /**
   * Sets up the kernel initially by passing the corresponding KernelOptions object
   * @param options KernelOptions
   */
  virtual void Setup(KernelOptions *options) = 0;

  /**
   * Initializes a state object
   * @param state
   */
  virtual void InitializeState(void *state) = 0;

  /**
   * Updates running state by aggregating the value
   * @param value
   * @param state
   */
  virtual void Update(const void *value, void *state) = 0;

  /**
   * Converts a state to the final result value
   * @param state
   * @param result
   */
  virtual void Finalize(const void *state, void *result) = 0;
};

/**
 * Adapter class that would apply type information to the AggregationKernel
 * @tparam OP_ID AggregationOpId
 * @tparam T
 * @tparam State - state type derived from KernelTraits
 * @tparam ResultT - result type derived from KernelTraits
 * @tparam Options - options type derived from KernelTraits
 */
template<AggregationOpId OP_ID, typename T,
    typename State = typename KernelTraits<OP_ID, T>::State,
    typename ResultT = typename KernelTraits<OP_ID, T>::ResultT,
    typename Options = typename KernelTraits<OP_ID, T>::Options>
struct TypedAggregationKernel : AggregationKernel {
  ~TypedAggregationKernel() override = default;
  virtual void Setup(Options *options) {};
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

/**
 * Mean kernel
 */
template<typename T>
class MeanKernel : public TypedAggregationKernel<AggregationOpId::MEAN, T> {
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

/**
 * Variance kernel
 */
template<typename T>
class VarianceKernel : public TypedAggregationKernel<AggregationOpId::VAR, T> {
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

/**
 * Sum kernel
 */
template<typename T>
struct SumKernel : public TypedAggregationKernel<AggregationOpId::SUM, T> {
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

/**
 * Count kernel
 */
template<typename T>
struct CountKernel : public TypedAggregationKernel<AggregationOpId::COUNT, T> {
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

/**
 * Min kernel
 */
template<typename T>
struct MinKernel : public TypedAggregationKernel<AggregationOpId::MIN, T> {
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

/**
 * Max kernel
 */
template<typename T>
struct MaxKernel : public TypedAggregationKernel<AggregationOpId::MAX, T> {
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

// -----------------------------------------------------------------------------

/**
 * Creates an AggregationKernel based on AggregationOpId
 * @tparam OP_ID AggregationOpId
 * @tparam T
 * @return
 */
template<AggregationOpId OP_ID, typename T>
std::unique_ptr<AggregationKernel> CreateAggregateKernel() {
  switch (OP_ID) {
    case SUM: return std::make_unique<SumKernel<T>>();
    case MIN:return std::make_unique<MinKernel<T>>();
    case MAX:return std::make_unique<MaxKernel<T>>();
    case COUNT:return std::make_unique<CountKernel<T>>();
    case MEAN:return std::make_unique<MeanKernel<T>>();
    case VAR:return std::make_unique<VarianceKernel<T>>();
  }
}

template<typename T>
std::unique_ptr<AggregationKernel> CreateAggregateKernel(AggregationOpId op_id) {
  switch (op_id) {
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
