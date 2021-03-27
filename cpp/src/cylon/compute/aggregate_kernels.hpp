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
#include <math.h>
#include <unordered_set>

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
  VAR,
  NUNIQUE,
  QUANTILE,
  STDDEV
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
  VarKernelOptions() : VarKernelOptions(1) {}

  int ddof;
};

struct QuantileKernelOptions : public KernelOptions {
  /**
   * @param quantile quantile
   */
  explicit QuantileKernelOptions(double quantile) : quantile(quantile) {}
  QuantileKernelOptions() : QuantileKernelOptions(0.5) {}

  double quantile;
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

  virtual ~AggregationOp() = default;

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
struct NUniqueOp : public BaseAggregationOp<NUNIQUE> {};

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

struct StdDevOp : public AggregationOp {
  explicit StdDevOp(int ddof) : AggregationOp(STDDEV, std::make_unique<VarKernelOptions>(ddof)) {}

  static inline std::unique_ptr<AggregationOp> Make(int ddof = 0) {
    return std::make_unique<StdDevOp>(ddof);
  }
};

/**
 * Var op
 */
struct QuantileOp : public AggregationOp {
  /**
   * @param quantile
   */
  explicit QuantileOp(double quantile) : AggregationOp(QUANTILE, std::make_unique<QuantileKernelOptions>(quantile)) {}

  static inline std::unique_ptr<AggregationOp> Make(double quantile = 0.5) {
    return std::make_unique<QuantileOp>(quantile);
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
struct KernelTraits<AggregationOpId::STDDEV, T> {
  using State = std::tuple<T, T, int64_t>; // <running sum of squares, running sum, running count>
  using ResultT = double_t;
  using Options = VarKernelOptions;
  static constexpr const char *name() { return "std_"; }
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

template<typename T>
struct KernelTraits<AggregationOpId::NUNIQUE, T> {
  using State = std::unordered_set<T>;
  using ResultT = int64_t;
  using Options = DefaultKernelOptions;
  static constexpr const char *name() { return "nunique_"; }
};

template<typename T>
struct KernelTraits<AggregationOpId::QUANTILE, T> {
  using State = std::vector<T>; // <running sum, running count>
  using ResultT = double_t;
  using Options = QuantileKernelOptions;
  static constexpr const char *name() { return "quantile_"; }
};

// -----------------------------------------------------------------------------

/**
 * 3. Aggregation Kernel
 */

struct AggregationKernel {
  virtual ~AggregationKernel() = default;

  /**
   * Sets up the kernel initially by passing the corresponding KernelOptions object
   * @param options KernelOptions
   */
  virtual void Setup(KernelOptions *options) = 0;

  /**
   * Initializes a state object
   * @param state
   * @param num_elements number of possible elements to aggregate
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
 * CRTP based AggregationKernel impl with types (for static polymorphism)
 * ref: https://www.modernescpp.com/index.php/c-is-still-lazy
 * @tparam DERIVED derived class
 * @tparam T
 * @tparam State - state type derived from KernelTraits
 * @tparam ResultT - result type derived from KernelTraits
 * @tparam Options - options type derived from KernelTraits
 *
 *  every derived class needs to implement the following methods
 *  1. void InitializeState(State *state)
 *  2. void Update(const T *value, State *state)
 *  3. void Finalize(const State *state, ResultT *result)
 *
 *  Optionally following setup method can be used to ingest kernel options,
 *  void Setup(Options *options)
 */
template<typename DERIVED, typename T, typename State, typename ResultT, typename Options>
struct TypedAggregationKernel : public AggregationKernel {
  // destructor for runtime polymorphism
  ~TypedAggregationKernel() override = default;

  // base setup implementation for CRTP (if derived classes have not implemented setup, this impl will be used!)
  void Setup(Options *options) {};

  inline void Setup(KernelOptions *options) override {
    return static_cast<DERIVED &>(*this).Setup(static_cast<Options *> (options));
  }

  inline void InitializeState(void *state) override {
    return static_cast<DERIVED &>(*this).InitializeState(static_cast<State *> (state));
  }

  inline void Update(const void *value, void *state) override {
    return static_cast<DERIVED &>(*this).Update(static_cast<const T *>(value), static_cast<State *>(state));
  };

  inline void Finalize(const void *state, void *result) override {
    return static_cast<DERIVED &>(*this).Finalize(static_cast<const State *>(state), static_cast<ResultT *>(result));
  }
};

/**
 * Mean kernel
 */
template<typename T>
class MeanKernel : public TypedAggregationKernel<MeanKernel<T>,
                                                 T,
                                                 typename KernelTraits<MEAN, T>::State,
                                                 typename KernelTraits<MEAN, T>::ResultT,
                                                 typename KernelTraits<MEAN, T>::Options> {
 public:
  void InitializeState(std::tuple<T, int64_t> *state) {
    *state = std::make_tuple(0, 0);
  }
  void Update(const T *value, std::tuple<T, int64_t> *state) {
    std::get<0>(*state) += *value;
    std::get<1>(*state) += 1;
  }
  void Finalize(const std::tuple<T, int64_t> *state, T *result) {
    if (std::get<1>(*state) != 0) {
      *result = std::get<0>(*state) / std::get<1>(*state);
    }
  }
};

/**
 * Variance kernel
 */
template<typename T>
class VarianceKernel : public TypedAggregationKernel<VarianceKernel<T>,
                                                     T,
                                                     typename KernelTraits<VAR, T>::State,
                                                     typename KernelTraits<VAR, T>::ResultT,
                                                     typename KernelTraits<VAR, T>::Options> {
 public:
  explicit VarianceKernel(bool do_std = false) : do_std(do_std) {}

  void Setup(VarKernelOptions *options) {
    ddof = options->ddof;
  }

  inline void InitializeState(std::tuple<T, T, int64_t> *state) {
    *state = std::make_tuple(static_cast<T>(0), static_cast<T>(0), 0);
  }
  inline void Update(const T *value, std::tuple<T, T, int64_t> *state) {
    std::get<0>(*state) += (*value) * (*value);
    std::get<1>(*state) += *value;
    std::get<2>(*state) += 1;
  }
  inline void Finalize(const std::tuple<T, T, int64_t> *state, double *result) {
    if (std::get<2>(*state) == 1) {
      *result = 0;
    } else if (std::get<2>(*state) != 0) {
      double mean = static_cast<double>(std::get<1>(*state)) / static_cast<double>(std::get<2>(*state));
      double mean_sum_sq = static_cast<double>(std::get<0>(*state)) / static_cast<double>(std::get<2>(*state));
      double
          var = static_cast<double>(std::get<2>(*state)) * (mean_sum_sq - mean * mean) / (std::get<2>(*state) - ddof);

      *result = do_std ? sqrt(var) : var;
    }
  };

 private:
  int ddof;
  bool do_std;
};

/**
 * Sum kernel
 */
template<typename T>
struct SumKernel : public TypedAggregationKernel<SumKernel<T>,
                                                 T,
                                                 typename KernelTraits<SUM, T>::State,
                                                 typename KernelTraits<SUM, T>::ResultT,
                                                 typename KernelTraits<SUM, T>::Options> {
  inline void InitializeState(std::tuple<T> *state) {
    *state = std::make_tuple(0);
  }
  inline void Update(const T *value, std::tuple<T> *state) {
    std::get<0>(*state) += *value;
  }
  inline void Finalize(const std::tuple<T> *state, T *result) {
    *result = std::get<0>(*state);
  }
};

/**
 * Count kernel
 */
template<typename T>
struct CountKernel : public TypedAggregationKernel<CountKernel<T>,
                                                   T,
                                                   typename KernelTraits<COUNT, T>::State,
                                                   typename KernelTraits<COUNT, T>::ResultT,
                                                   typename KernelTraits<COUNT, T>::Options> {
  inline void InitializeState(std::tuple<int64_t> *state) {
    *state = std::make_tuple(0);
  }
  inline void Update(const T *value, std::tuple<int64_t> *state) {
    std::get<0>(*state) += 1;
  }
  inline void Finalize(const std::tuple<int64_t> *state, long *result) {
    *result = std::get<0>(*state);
  }
};

/**
 * Min kernel
 */
template<typename T>
struct MinKernel : public TypedAggregationKernel<MinKernel<T>,
                                                 T,
                                                 typename KernelTraits<MIN, T>::State,
                                                 typename KernelTraits<MIN, T>::ResultT,
                                                 typename KernelTraits<MIN, T>::Options> {
  inline void InitializeState(std::tuple<T> *state) {
    *state = std::make_tuple(std::numeric_limits<T>::max());
  }
  inline void Update(const T *value, std::tuple<T> *state) {
    std::get<0>(*state) = std::min(*value, std::get<0>(*state));
  }
  inline void Finalize(const std::tuple<T> *state, T *result) {
    *result = std::get<0>(*state);
  }
};

/**
 * Max kernel
 */
template<typename T>
struct MaxKernel : public TypedAggregationKernel<MaxKernel<T>,
                                                 T,
                                                 typename KernelTraits<MAX, T>::State,
                                                 typename KernelTraits<MAX, T>::ResultT,
                                                 typename KernelTraits<MAX, T>::Options> {
  inline void InitializeState(std::tuple<T> *state) {
    *state = std::make_tuple(std::numeric_limits<T>::min());
  }
  inline void Update(const T *value, std::tuple<T> *state) {
    std::get<0>(*state) = std::max(*value, std::get<0>(*state));
  }
  inline void Finalize(const std::tuple<T> *state, T *result) {
    *result = std::get<0>(*state);
  }
};

template<typename T>
struct NUniqueKernel : public TypedAggregationKernel<NUniqueKernel<T>,
                                                     T,
                                                     typename KernelTraits<NUNIQUE, T>::State,
                                                     typename KernelTraits<NUNIQUE, T>::ResultT,
                                                     typename KernelTraits<NUNIQUE, T>::Options> {
  inline void InitializeState(std::unordered_set<T> *state) {}
  inline void Update(const T *value, std::unordered_set<T> *state) {
    state->emplace(*value);
  }
  inline void Finalize(const std::unordered_set<T> *state, long *result) {
    *result = state->size();
    const_cast<std::unordered_set<T> *>(state)->clear();
  }
};

template<typename T>
struct QuantileKernel : public TypedAggregationKernel<QuantileKernel<T>,
                                                      T,
                                                      typename KernelTraits<QUANTILE, T>::State,
                                                      typename KernelTraits<QUANTILE, T>::ResultT,
                                                      typename KernelTraits<QUANTILE, T>::Options> {
  inline void Setup(QuantileKernelOptions *options) {
    quantile = options->quantile;
  }
  inline void InitializeState(std::vector<T> *state) {}
  inline void Update(const T *value, std::vector<T> *state) {
    state->emplace_back(*value);
  }
  inline void Finalize(const std::vector<T> *state, double *result) {
    // ref: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/quantile using type 2
    double np = state->size() * quantile, j = floor(np), g = np - j;
    const auto pos = static_cast<size_t>(j);

    auto *mutable_state = const_cast<std::vector<T> *>(state); // make state mutable

    // partition vector around pos'th element
    std::nth_element(mutable_state->begin(), mutable_state->begin() + pos, mutable_state->end());

    if (g == 0) {
      *result =
          0.5 * (*std::max_element(mutable_state->begin(), mutable_state->begin() + pos) + mutable_state->at(pos));
    } else {
      *result = static_cast<double>(mutable_state->at(pos));
    }

    mutable_state->clear();
  }

  double quantile;
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
    case NUNIQUE:return std::make_unique<NUniqueKernel<T>>();
    case QUANTILE:return std::make_unique<QuantileKernel<T>>();
    case STDDEV: return std::make_unique<VarianceKernel<T>>(true);
    default:return nullptr;
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
    case NUNIQUE:return std::make_unique<NUniqueKernel<T>>();
    case QUANTILE:return std::make_unique<QuantileKernel<T>>();
    case STDDEV: return std::make_unique<VarianceKernel<T>>(true);
    default:return nullptr;
  }
}

}
}

#endif //CYLON_CPP_SRC_CYLON_COMPUTE_AGGREGATE_KERNELS_HPP_
