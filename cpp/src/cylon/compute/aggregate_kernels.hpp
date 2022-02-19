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

#include <algorithm>
#include <cmath>
#include <vector>
#include <unordered_set>

#include "cylon/util/macros.hpp"

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
//struct AggregationOp {
//  /**
//   * @param id AggregationOpId
//   * @param options unique_ptr of options
//   */
//  AggregationOp(AggregationOpId id, KernelOptions *options)
//      : id(id), options(options) {}
//
//  /**
//   * Constructor with uninitialized options. This would be explicitly handled in impl
//   * @param id
//   */
//  explicit AggregationOp(AggregationOpId id) : id(id), options(nullptr) {}
//
//  virtual ~AggregationOp() = default;
//
//  AggregationOpId id;
//  KernelOptions* options;
//};

struct AggregationOp {
  virtual ~AggregationOp() = default;
  virtual AggregationOpId id() const = 0;
  virtual KernelOptions *options() const { return nullptr; };
};

/**
 * Base aggregation operation with KernelOptions
 * @tparam ID AggregationOpId
 */
template<AggregationOpId ID>
struct BaseAggregationOp : public AggregationOp {
  AggregationOpId id() const override { return ID; }

  static std::shared_ptr<AggregationOp> Make() {
    return std::make_shared<BaseAggregationOp<ID>>();
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
  std::shared_ptr<VarKernelOptions> opt;

  /**
   * @param ddof delta degree of freedom
   */
  explicit VarOp(int ddof = 1) : opt(std::make_shared<VarKernelOptions>(ddof)) {}
  explicit VarOp(const std::shared_ptr<KernelOptions> &opt)
      : opt(std::static_pointer_cast<VarKernelOptions>(opt)) {}

  AggregationOpId id() const override { return VAR; }
  KernelOptions *options() const override { return opt.get(); }

  static std::shared_ptr<AggregationOp> Make(int ddof = 1) {
    return std::make_shared<VarOp>(ddof);
  }
};

struct StdDevOp : public VarOp {
  explicit StdDevOp(int ddof = 1) : VarOp(ddof) {}
  explicit StdDevOp(const std::shared_ptr<KernelOptions> &opt) : VarOp(opt) {}
  AggregationOpId id() const override { return STDDEV; }

  static std::shared_ptr<AggregationOp> Make(int ddof = 1) {
    return std::make_shared<StdDevOp>(ddof);
  }
};

/**
 * Var op
 */
struct QuantileOp : public AggregationOp {
  std::shared_ptr<QuantileKernelOptions> opt;

  /**
   * @param quantile
   */
  explicit QuantileOp(double quantile = 0.5)
      : opt(std::make_shared<QuantileKernelOptions>(quantile)) {}
  explicit QuantileOp(const std::shared_ptr<KernelOptions> &opt)
      : opt(std::static_pointer_cast<QuantileKernelOptions>(opt)) {}

  AggregationOpId id() const override { return QUANTILE; }
  KernelOptions *options() const override { return opt.get(); }

  static std::shared_ptr<AggregationOp> Make(double quantile = 0.5) {
    return std::make_shared<QuantileOp>(quantile);
  }
};

std::shared_ptr<AggregationOp> MakeAggregationOpFromID(AggregationOpId id);

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
  using Options = KernelOptions;
  static constexpr const char *name() { return "sum_"; }

};

template<typename T>
struct KernelTraits<AggregationOpId::MEAN, T> {
  using State = std::tuple<T, int64_t>; // <running sum, running count>
  using ResultT = T;
  using Options = KernelOptions;
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
  using Options = KernelOptions;
  static constexpr const char *name() { return "count_"; }
};

template<typename T>
struct KernelTraits<AggregationOpId::MIN, T> {
  using State = std::tuple<T>;
  using ResultT = T;
  using Options = KernelOptions;
  static constexpr const char *name() { return "min_"; }
};

template<typename T>
struct KernelTraits<AggregationOpId::MAX, T> {
  using State = std::tuple<T>;
  using ResultT = T;
  using Options = KernelOptions;
  static constexpr const char *name() { return "max_"; }
};

template<typename T>
struct KernelTraits<AggregationOpId::NUNIQUE, T> {
  using State = std::unordered_set<T>;
  using ResultT = int64_t;
  using Options = KernelOptions;
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
  virtual void InitializeState(void *state) const = 0;

  /**
   * Updates running state by aggregating the value
   * @param value
   * @param state
   */
  virtual void Update(const void *value, void *state) const = 0;

  /**
   * Converts a state to the final result value
   * @param state
   * @param result
   */
  virtual void Finalize(const void *state, void *result) const = 0;
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
template<typename DERIVED, AggregationOpId opp_id, typename T>
struct TypedAggregationKernel : public AggregationKernel {
  using State = typename KernelTraits<opp_id, T>::State;
  using ResultT = typename KernelTraits<opp_id, T>::ResultT;
  using Options = typename KernelTraits<opp_id, T>::Options;

  // destructor for runtime polymorphism
  ~TypedAggregationKernel() override = default;

  inline void Setup(KernelOptions *options) override {
    return static_cast<DERIVED &>(*this).KernelSetup(static_cast<Options *> (options));
  }

  inline void InitializeState(void *state) const override {
    return static_cast<const DERIVED &>(*this).KernelInitializeState(static_cast<State *> (state));
  }

  inline void Update(const void *value, void *state) const override {
    return static_cast<const DERIVED &>(*this)
        .KernelUpdate(static_cast<const T *>(value), static_cast<State *>(state));
  };

  inline void Finalize(const void *state, void *result) const override {
    return static_cast<const DERIVED &>(*this)
        .KernelFinalize(static_cast<const State *>(state), static_cast<ResultT *>(result));
  }
};

/**
 * Mean kernel
 */
template<typename T>
class MeanKernel : public TypedAggregationKernel<MeanKernel<T>, MEAN, T> {
 public:
  void KernelSetup(KernelOptions *options) {
    CYLON_UNUSED(options);
  };

  void KernelInitializeState(std::tuple<T, int64_t> *state) const {
    *state = std::make_tuple(0, 0);
  }

  void KernelUpdate(const T *value, std::tuple<T, int64_t> *state) const {
    std::get<0>(*state) += *value;
    std::get<1>(*state) += 1;
  }
  void KernelFinalize(const std::tuple<T, int64_t> *state, T *result) const {
    if (std::get<1>(*state) != 0) {
      *result = std::get<0>(*state) / std::get<1>(*state);
    }
  }
};

/**
 * Variance kernel
 */
template<typename T>
class VarianceKernel : public TypedAggregationKernel<VarianceKernel<T>, VAR, T> {
 public:
  explicit VarianceKernel(bool do_std = false) : ddof(0), do_std(do_std) {}

  void KernelSetup(VarKernelOptions *options) {
    ddof = options->ddof;
  }

  inline void KernelInitializeState(std::tuple<T, T, int64_t> *state) const {
    *state = std::make_tuple(static_cast<T>(0), static_cast<T>(0), 0);
  }
  inline void KernelUpdate(const T *value, std::tuple<T, T, int64_t> *state) const {
    std::get<0>(*state) += (*value) * (*value);
    std::get<1>(*state) += *value;
    std::get<2>(*state) += 1;
  }
  inline void KernelFinalize(const std::tuple<T, T, int64_t> *state, double_t *result) const {
    if (std::get<2>(*state) == 1) {
      *result = 0;
    } else if (std::get<2>(*state) != 0) {
      double mean =
          static_cast<double>(std::get<1>(*state)) / static_cast<double>(std::get<2>(*state));
      double mean_sum_sq =
          static_cast<double>(std::get<0>(*state)) / static_cast<double>(std::get<2>(*state));
      double
          var = static_cast<double>(std::get<2>(*state)) * (mean_sum_sq - mean * mean)
          / (std::get<2>(*state) - ddof);

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
struct SumKernel : public TypedAggregationKernel<SumKernel<T>, SUM, T> {
  void KernelSetup(KernelOptions *options) {
    CYLON_UNUSED(options);
  };
  inline void KernelInitializeState(std::tuple<T> *state) const {
    *state = std::make_tuple(0);
  }
  inline void KernelUpdate(const T *value, std::tuple<T> *state) const {
    std::get<0>(*state) += *value;
  }
  inline void KernelFinalize(const std::tuple<T> *state, T *result) const {
    *result = std::get<0>(*state);
  }
};

/**
 * Count kernel
 */
template<typename T>
struct CountKernel : public TypedAggregationKernel<CountKernel<T>, COUNT, T> {
  void KernelSetup(KernelOptions *options) {
    CYLON_UNUSED(options);
  };
  inline void KernelInitializeState(std::tuple<int64_t> *state) const {
    *state = std::make_tuple(0);
  }
  inline void KernelUpdate(const T *value, std::tuple<int64_t> *state) const {
    CYLON_UNUSED(value);
    std::get<0>(*state) += 1;
  }
  inline void KernelFinalize(const std::tuple<int64_t> *state, int64_t *result) const {
    *result = std::get<0>(*state);
  }
};

/**
 * Min kernel
 */
template<typename T>
struct MinKernel : public TypedAggregationKernel<MinKernel<T>, MIN, T> {
  void KernelSetup(KernelOptions *options) {
    CYLON_UNUSED(options);
  };
  inline void KernelInitializeState(std::tuple<T> *state) const {
    *state = std::make_tuple(std::numeric_limits<T>::max());
  }
  inline void KernelUpdate(const T *value, std::tuple<T> *state) const {
    std::get<0>(*state) = std::min(*value, std::get<0>(*state));
  }
  inline void KernelFinalize(const std::tuple<T> *state, T *result) const {
    *result = std::get<0>(*state);
  }
};

/**
 * Max kernel
 */
template<typename T>
struct MaxKernel : public TypedAggregationKernel<MaxKernel<T>, MAX, T> {
  void KernelSetup(KernelOptions *options) {
    CYLON_UNUSED(options);
  };
  inline void KernelInitializeState(std::tuple<T> *state) const {
    *state = std::make_tuple(std::numeric_limits<T>::min());
  }
  inline void KernelUpdate(const T *value, std::tuple<T> *state) const {
    std::get<0>(*state) = std::max(*value, std::get<0>(*state));
  }
  inline void KernelFinalize(const std::tuple<T> *state, T *result) const {
    *result = std::get<0>(*state);
  }
};

template<typename T>
struct NUniqueKernel : public TypedAggregationKernel<NUniqueKernel<T>, NUNIQUE, T> {
  void KernelSetup(KernelOptions *options) {
    CYLON_UNUSED(options);
  };
  inline void KernelInitializeState(std::unordered_set<T> *state) const { CYLON_UNUSED(state); }
  inline void KernelUpdate(const T *value, std::unordered_set<T> *state) const {
    state->emplace(*value);
  }
  inline void KernelFinalize(const std::unordered_set<T> *state, int64_t *result) const {
    *result = state->size();
    const_cast<std::unordered_set<T> *>(state)->clear();
  }
};

template<typename T>
struct QuantileKernel : public TypedAggregationKernel<QuantileKernel<T>, QUANTILE, T> {
  inline void KernelSetup(QuantileKernelOptions *options) {
    quantile = options->quantile;
  }
  inline void KernelInitializeState(std::vector<T> *state) const {
    CYLON_UNUSED(state);
  }
  inline void KernelUpdate(const T *value, std::vector<T> *state) const {
    state->emplace_back(*value);
  }
  inline void KernelFinalize(const std::vector<T> *state, double_t *result) const {
    // ref: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/quantile using type 2
    double np = state->size() * quantile, j = floor(np), g = np - j;
    const auto pos = static_cast<size_t>(j);

    auto *mutable_state = const_cast<std::vector<T> *>(state); // make state mutable

    // partition vector around pos'th element
    std::nth_element(mutable_state->begin(), mutable_state->begin() + pos, mutable_state->end());

    if (g == 0) {
      *result =
          0.5 * (*std::max_element(mutable_state->begin(), mutable_state->begin() + pos)
              + mutable_state->at(pos));
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
