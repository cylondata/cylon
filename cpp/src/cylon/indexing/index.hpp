#ifndef CYLON_SRC_CYLON_INDEXING_INDEX_H_
#define CYLON_SRC_CYLON_INDEXING_INDEX_H_

#include "index.hpp"

#include "status.hpp"
#include "ctx/cylon_context.hpp"
#include "ctx/arrow_memory_pool_utils.hpp"
#include "util/macros.hpp"
#include "util/arrow_utils.hpp"
#include <glog/logging.h>
#include <arrow/table.h>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/kernel.h>
#include <arrow/arrow_comparator.hpp>
#include "thridparty/flat_hash_map/unordered_map.hpp"
#include <chrono>

namespace cylon {

enum IndexingSchema {
  Range = 0,
  Linear = 1,
  Hash = 2,
  BinaryTree = 3,
  BTree = 4,
};

class BaseIndex {

 public:

  explicit BaseIndex(int col_id, int size, std::shared_ptr<CylonContext> &ctx) : size_(size), col_id_(col_id) {
    pool_ = cylon::ToArrowPool(ctx);
  };

  explicit BaseIndex(int col_id, int size, arrow::MemoryPool *pool) {
    col_id_ = col_id;
    size_ = size;
    pool_ = pool;
  };

  // TODO: virtual destructor
  virtual Status LocationByValue(const void *search_param,
                                 std::vector<int64_t> &find_index) = 0;

  virtual Status LocationByValue(const void *search_param, int64_t &find_index) = 0;

  virtual Status LocationByValue(const void *search_param,
                                 const std::shared_ptr<arrow::Table> &input,
                                 std::vector<int64_t> &filter_location,
                                 std::shared_ptr<arrow::Table> &output) = 0;

  virtual std::shared_ptr<arrow::Array> GetIndexAsArray() = 0;

  virtual void SetIndexArray(std::shared_ptr<arrow::Array> &index_arr) = 0;

  virtual std::shared_ptr<arrow::Array> GetIndexArray() = 0;

  virtual int GetColId() const;

  virtual int GetSize() const;

  virtual IndexingSchema GetSchema() = 0;

  virtual arrow::MemoryPool *GetPool() const;

  virtual bool IsUnique() = 0;

 private:
  int size_;
  int col_id_;
  arrow::MemoryPool *pool_;
  std::shared_ptr<arrow::Array> index_arr;

};

cylon::Status CompareArraysForUniqueness(std::shared_ptr<arrow::Array> &index_arr, bool &is_unique);

template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
class HashIndex : public BaseIndex {
 public:
  using MMAP_TYPE = typename std::unordered_multimap<CTYPE, int64_t>;
  HashIndex(int col_ids, int size, arrow::MemoryPool *pool, std::shared_ptr<MMAP_TYPE> map)
      : BaseIndex(col_ids, size, pool) {
    map_ = map;
  };

  Status LocationByValue(const void *search_param,
                         const std::shared_ptr<arrow::Table> &input,
                         std::vector<int64_t> &filter_locations,
                         std::shared_ptr<arrow::Table> &output) override {

    arrow::Status arrow_status;
    cylon::Status status;
    std::shared_ptr<arrow::Array> out_idx;
    arrow::compute::ExecContext fn_ctx(GetPool());
    arrow::Int64Builder idx_builder(GetPool());
    const arrow::Datum input_table(input);

    status = LocationByValue(search_param, filter_locations);

    if (!status.is_ok()) {
      LOG(ERROR) << "Error occurred in obtaining filter locations by index value";
      return status;
    }

    arrow_status = idx_builder.AppendValues(filter_locations);

    if (!arrow_status.ok()) {
      LOG(ERROR) << "Error occurred in appending filter indices to builder";
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
    }

    arrow_status = idx_builder.Finish(&out_idx);

    if (!arrow_status.ok()) {
      LOG(ERROR) << "Error occurred in builder finish";
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
    }

    const arrow::Datum filter_indices(out_idx);
    arrow::Result<arrow::Datum>
        result = arrow::compute::Take(input_table, filter_indices, arrow::compute::TakeOptions::Defaults(), &fn_ctx);

    if (!result.status().ok()) {
      LOG(ERROR) << "Error occurred in filtering table by indices";
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
    }

    output = result.ValueOrDie().table();
    return Status::OK();
  }

  Status LocationByValue(const void *search_param, std::vector<int64_t> &find_index) override {
    const CTYPE val = *static_cast<const CTYPE *>(search_param);
    auto ret = map_->equal_range(val);
    for (auto it = ret.first; it != ret.second; ++it) {
      find_index.push_back(it->second);
    }
    return Status::OK();
  }

  Status LocationByValue(const void *search_param, int64_t &find_index) override {
    const CTYPE val = *static_cast<const CTYPE *>(search_param);
    auto ret = map_->find(val);
    if (ret != map_->end()) {
      find_index = ret->second;
      return Status::OK();
    }
    return Status(cylon::Code::IndexError);
  }

  std::shared_ptr<arrow::Array> GetIndexAsArray() override {

    using ARROW_ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;
    using ARROW_BUILDER_T = typename arrow::TypeTraits<ARROW_T>::BuilderType;

    arrow::Status arrow_status;
    auto pool = GetPool();

    ARROW_BUILDER_T builder(this->GetIndexArray()->type(), pool);

    std::shared_ptr<ARROW_ARRAY_T> index_array;

    std::vector<CTYPE> vec(GetSize(), 1);

    for (const auto &x: *map_) {
      vec[x.second] = x.first;
    }

    builder.AppendValues(vec);
    arrow_status = builder.Finish(&index_array);

    if (!arrow_status.ok()) {
      LOG(ERROR) << "Error occurred in retrieving index";
      return nullptr;
    }

    return index_array;
  }

  int GetColId() const override {
    return BaseIndex::GetColId();
  }
  int GetSize() const override {
    return BaseIndex::GetSize();
  }
  arrow::MemoryPool *GetPool() const override {
    return BaseIndex::GetPool();
  }

  void SetIndexArray(std::shared_ptr<arrow::Array> &index_arr) override {
    index_arr_ = index_arr;
  }

  std::shared_ptr<arrow::Array> GetIndexArray() override {
    return index_arr_;
  }

  bool IsUnique() override {
    bool is_unique = false;
    auto index_arr = GetIndexArray();
    auto status = CompareArraysForUniqueness(index_arr, is_unique);
    if (!status.is_ok()) {
      LOG(ERROR) << "Error occurred in is unique operation";
    }
    return is_unique;
  }

  IndexingSchema GetSchema() override {
    return IndexingSchema::Hash;
  }

 private:
  std::shared_ptr<MMAP_TYPE> map_;
  std::shared_ptr<arrow::Array> index_arr_;
};

template<>
class HashIndex<arrow::StringType, arrow::util::string_view> : public BaseIndex {
 public:
  using MMAP_TYPE = typename std::unordered_multimap<arrow::util::string_view, int64_t>;
  HashIndex(int col_ids, int size, arrow::MemoryPool *pool, std::shared_ptr<MMAP_TYPE> map)
      : BaseIndex(col_ids, size, pool) {
    map_ = map;
  };

  Status LocationByValue(const void *search_param,
                         const std::shared_ptr<arrow::Table> &input,
                         std::vector<int64_t> &filter_locations,
                         std::shared_ptr<arrow::Table> &output) override {
    LOG(INFO) << "Extract table for a given index";
    arrow::Status arrow_status;
    cylon::Status status;
    std::shared_ptr<arrow::Array> out_idx;
    arrow::compute::ExecContext fn_ctx(GetPool());
    arrow::Int64Builder idx_builder(GetPool());
    const arrow::Datum input_table(input);

    status = LocationByValue(search_param, filter_locations);

    if (!status.is_ok()) {
      LOG(ERROR) << "Error occurred in filtering indices by index value";
      return status;
    }

    arrow_status = idx_builder.AppendValues(filter_locations);

    if (!arrow_status.ok()) {
      LOG(ERROR) << "Error occurred in appending indices to builder";
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
    }

    arrow_status = idx_builder.Finish(&out_idx);

    if (!arrow_status.ok()) {
      LOG(ERROR) << "Error occurred in builder finish";
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
    }

    const arrow::Datum filter_indices(out_idx);
    arrow::Result<arrow::Datum>
        result = arrow::compute::Take(input_table, filter_indices, arrow::compute::TakeOptions::Defaults(), &fn_ctx);
    if (!result.status().ok()) {
      LOG(ERROR) << "Error occurred in filtering table by indices";
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
    }
    output = result.ValueOrDie().table();
    return Status::OK();
  };

  Status LocationByValue(const void *search_param, std::vector<int64_t> &find_index) override {
    LOG(INFO) << "Finding row ids for a given index";
    const std::string *sp = static_cast<const std::string *>(search_param);
    arrow::util::string_view search_param_sv(*sp);
    auto ret = map_->equal_range(search_param_sv);
    for (auto it = ret.first; it != ret.second; ++it) {
      find_index.push_back(it->second);
    }
    return Status::OK();
  };

  Status LocationByValue(const void *search_param, int64_t &find_index) override {
    LOG(INFO) << "Finding row id for a given index";
    const std::string *sp = static_cast<const std::string *>(search_param);
    arrow::util::string_view search_param_sv(*sp);
    auto ret = map_->find(search_param_sv);
    if (ret != map_->end()) {
      find_index = ret->second;
      return Status::OK();
    }
    return Status(cylon::Code::IndexError);
  };

  std::shared_ptr<arrow::Array> GetIndexAsArray() override {

    arrow::Status arrow_status;
    auto pool = GetPool();

    arrow::StringBuilder builder(pool);

    std::shared_ptr<arrow::StringArray> index_array;

    std::vector<std::string> vec(GetSize(), "");

    for (const auto &x: *map_) {
      vec[x.second] = x.first.to_string();
    }
    std::cout << std::endl;

    builder.AppendValues(vec);
    arrow_status = builder.Finish(&index_array);

    if (!arrow_status.ok()) {
      LOG(ERROR) << "Error occurred in retrieving index";
      return nullptr;
    }

    return index_array;
  }

  int GetColId() const override {
    return BaseIndex::GetColId();
  }
  int GetSize() const override {
    return BaseIndex::GetSize();
  }
  arrow::MemoryPool *GetPool() const override {
    return BaseIndex::GetPool();
  }

  void SetIndexArray(std::shared_ptr<arrow::Array> &index_arr) override {
    index_arr_ = index_arr;
  }

  std::shared_ptr<arrow::Array> GetIndexArray() override {
    return index_arr_;
  }

  bool IsUnique() override {
    bool is_unique = false;
    auto index_arr = GetIndexArray();
    auto status = CompareArraysForUniqueness(index_arr, is_unique);
    if (!status.is_ok()) {
      LOG(ERROR) << "Error occurred in is unique operation";
    }
    return is_unique;
  }

  IndexingSchema GetSchema() override {
    return IndexingSchema::Hash;
  }


 private:
  std::shared_ptr<MMAP_TYPE> map_;
  std::shared_ptr<arrow::Array> index_arr_;
};

class RangeIndex : public BaseIndex {
 public:
  RangeIndex(int start, int size, int step, arrow::MemoryPool *pool);
  Status LocationByValue(const void *search_param,
                         const std::shared_ptr<arrow::Table> &input,
                         std::vector<int64_t> &filter_locations,
                         std::shared_ptr<arrow::Table> &output) override;
  Status LocationByValue(const void *search_param, std::vector<int64_t> &find_index) override;
  Status LocationByValue(const void *search_param, int64_t &find_index) override;
  std::shared_ptr<arrow::Array> GetIndexAsArray() override;
  void SetIndexArray(std::shared_ptr<arrow::Array> &index_arr) override;
  std::shared_ptr<arrow::Array> GetIndexArray() override;
  bool IsUnique() override;

  int GetColId() const override;
  int GetSize() const override;
  arrow::MemoryPool *GetPool() const override;

  int GetStart() const;
  int GetAnEnd() const;
  int GetStep() const;

  IndexingSchema GetSchema() override;

 private:
  int start_ = 0;
  int end_ = 0;
  int step_ = 1;
  std::shared_ptr<arrow::Array> index_arr_ = nullptr;

};

template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
class LinearIndex : public BaseIndex {
 public:
  using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  LinearIndex(int col_id, int size, arrow::MemoryPool *pool, std::shared_ptr<ARROW_ARRAY_TYPE> &index_array)
      : BaseIndex(col_id, size, pool), index_array_(index_array) {
  }

  Status LocationByValue(const void *search_param, std::vector<int64_t> &find_index) override {
    const CTYPE search_val = *static_cast<const CTYPE *>(search_param);
    for (int64_t ix = 0; ix < index_array_->length(); ix++) {
      CTYPE val = index_array_->GetView(ix);
      if (search_val == val) {
        find_index.push_back(ix);
      }
    }
    return Status::OK();
  }

  Status LocationByValue(const void *search_param, int64_t &find_index) override {
    const CTYPE search_val = *static_cast<const CTYPE *>(search_param);
    for (int64_t ix = 0; ix < index_array_->length(); ix++) {
      CTYPE val = index_array_->GetView(ix);
      if (search_val == val) {
        find_index = ix;
        break;
      }
    }
    return Status::OK();
  }
  Status LocationByValue(const void *search_param,
                         const std::shared_ptr<arrow::Table> &input,
                         std::vector<int64_t> &filter_location,
                         std::shared_ptr<arrow::Table> &output) override {
    arrow::Status arrow_status;
    cylon::Status status;
    std::shared_ptr<arrow::Array> out_idx;
    arrow::compute::ExecContext fn_ctx(GetPool());
    arrow::Int64Builder idx_builder(GetPool());
    const arrow::Datum input_table(input);

    status = LocationByValue(search_param, filter_location);

    if (!status.is_ok()) {
      LOG(ERROR) << "Error occurred in obtaining filter indices by index value";
      return status;
    }

    arrow_status = idx_builder.AppendValues(filter_location);

    if (!arrow_status.ok()) {
      LOG(ERROR) << "Error occurred in appending filter indices to builder";
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
    }

    arrow_status = idx_builder.Finish(&out_idx);

    if (!arrow_status.ok()) {
      LOG(ERROR) << "Error occurred in builder finish";
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
    }

    const arrow::Datum filter_indices(out_idx);
    arrow::Result<arrow::Datum>
        result = arrow::compute::Take(input_table, filter_indices, arrow::compute::TakeOptions::Defaults(), &fn_ctx);

    if (!result.status().ok()) {
      LOG(ERROR) << "Error occurred in filtering table by indices";
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
    }

    output = result.ValueOrDie().table();
    return Status::OK();
  }
  std::shared_ptr<arrow::Array> GetIndexAsArray() override {
    // TODO: determine to keep or remove
    return index_array_;
  }
  void SetIndexArray(std::shared_ptr<arrow::Array> &index_arr) override {
    index_array_ = std::static_pointer_cast<ARROW_ARRAY_TYPE>(index_arr);
  }
  std::shared_ptr<arrow::Array> GetIndexArray() override {
    return index_array_;
  }
  int GetColId() const override {
    return BaseIndex::GetColId();
  }
  int GetSize() const override {
    return BaseIndex::GetSize();
  }
  arrow::MemoryPool *GetPool() const override {
    return BaseIndex::GetPool();
  }

  bool IsUnique() override {
    bool is_unique = false;
    auto index_arr = GetIndexArray();
    auto status = CompareArraysForUniqueness(index_arr, is_unique);
    if (!status.is_ok()) {
      LOG(ERROR) << "Error occurred in is unique operation";
    }
    return is_unique;
  }

  IndexingSchema GetSchema() override {
    return Linear;
  }

 private:
  std::shared_ptr<ARROW_ARRAY_TYPE> index_array_;

};

template<>
class LinearIndex<arrow::StringType, arrow::util::string_view> : public BaseIndex {
 public:
  LinearIndex(int col_id, int size, arrow::MemoryPool *pool, std::shared_ptr<arrow::StringArray> &index_array)
      : BaseIndex(col_id, size, pool), index_array_(index_array) {
  }

  Status LocationByValue(const void *search_param, std::vector<int64_t> &find_index) override {
    const std::string &sp = *(static_cast<const std::string *>(search_param));
    arrow::util::string_view search_param_sv(sp);
    for (int64_t ix = 0; ix < index_array_->length(); ix++) {
      arrow::util::string_view val = index_array_->GetView(ix);
      if (search_param_sv == val) {
        find_index.push_back(ix);
      }
    }
    return Status::OK();
  }
  Status LocationByValue(const void *search_param, int64_t &find_index) override {
    const std::string *sp = static_cast<const std::string *>(search_param);
    arrow::util::string_view search_param_sv(*sp);
    for (int64_t ix = 0; ix < index_array_->length(); ix++) {
      arrow::util::string_view val = index_array_->GetView(ix);
      if (search_param_sv == val) {
        find_index = ix;
        break;
      }
    }
    return Status::OK();
  }
  Status LocationByValue(const void *search_param,
                         const std::shared_ptr<arrow::Table> &input,
                         std::vector<int64_t> &filter_location,
                         std::shared_ptr<arrow::Table> &output) override {
    LOG(INFO) << "Extract table for a given index";
    arrow::Status arrow_status;
    cylon::Status status;
    std::shared_ptr<arrow::Array> out_idx;
    arrow::compute::ExecContext fn_ctx(GetPool());
    arrow::Int64Builder idx_builder(GetPool());
    const arrow::Datum input_table(input);

    status = LocationByValue(search_param, filter_location);
    if (!status.is_ok()) {
      LOG(ERROR) << "Error occurred in retrieving location indices by index value";
      return status;
    }

    arrow_status = idx_builder.AppendValues(filter_location);

    if (!arrow_status.ok()) {
      LOG(ERROR) << "Error occurred in appending filter locations to builder";
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
    }

    arrow_status = idx_builder.Finish(&out_idx);

    if (!arrow_status.ok()) {
      LOG(ERROR) << "Error occurred in builder finish";
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
    }

    const arrow::Datum filter_indices(out_idx);
    arrow::Result<arrow::Datum>
        result = arrow::compute::Take(input_table, filter_indices, arrow::compute::TakeOptions::Defaults(), &fn_ctx);

    if (!result.status().ok()) {
      LOG(ERROR) << "Error occurred in filtering table by indices";
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
    }

    output = result.ValueOrDie().table();
    return Status::OK();
  }
  std::shared_ptr<arrow::Array> GetIndexAsArray() override {
    // TODO: determine to keep or remove
    return index_array_;
  }
  void SetIndexArray(std::shared_ptr<arrow::Array> &index_arr) override {
    index_array_ = std::static_pointer_cast<arrow::StringArray>(index_arr);
  }
  std::shared_ptr<arrow::Array> GetIndexArray() override {
    return index_array_;
  }
  int GetColId() const override {
    return BaseIndex::GetColId();
  }
  int GetSize() const override {
    return BaseIndex::GetSize();
  }
  arrow::MemoryPool *GetPool() const override {
    return BaseIndex::GetPool();
  }

  bool IsUnique() override {
    bool is_unique = false;
    auto index_arr = GetIndexArray();
    auto status = CompareArraysForUniqueness(index_arr, is_unique);
    if (!status.is_ok()) {
      LOG(ERROR) << "Error occurred in is unique operation";
    }
    return is_unique;
  }

  IndexingSchema GetSchema() override {
    return Linear;
  }

 private:
  std::shared_ptr<arrow::StringArray> index_array_;

};

class IndexKernel {
 public:
  explicit IndexKernel() {

  }
  virtual std::shared_ptr<BaseIndex> BuildIndex(arrow::MemoryPool *pool,
                                                std::shared_ptr<arrow::Table> &input_table,
                                                const int index_column) = 0;
};

class RangeIndexKernel : public IndexKernel {
 public:
  RangeIndexKernel();

  std::shared_ptr<BaseIndex> BuildIndex(arrow::MemoryPool *pool,
                                        std::shared_ptr<arrow::Table> &input_table,
                                        const int index_column) override;

};

template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
class HashIndexKernel : public IndexKernel {

 public:
  explicit HashIndexKernel() : IndexKernel() {}
  using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using MMAP_TYPE = typename std::unordered_multimap<CTYPE, int64_t>;

  std::shared_ptr<BaseIndex> BuildIndex(arrow::MemoryPool *pool,
                                        std::shared_ptr<arrow::Table> &input_table,
                                        const int index_column) override {

    const std::shared_ptr<arrow::Array> &idx_column = cylon::util::GetChunkOrEmptyArray(input_table->column(index_column), 0);
    std::shared_ptr<MMAP_TYPE> out_umm_ptr = std::make_shared<MMAP_TYPE>(idx_column->length());
    auto reader0 = std::static_pointer_cast<ARROW_ARRAY_TYPE>(idx_column);
    auto start_start = std::chrono::steady_clock::now();
    for (int64_t i = reader0->length() - 1; i >= 0; --i) {
      auto val = reader0->GetView(i);
      out_umm_ptr->emplace(val, i);
    }
    auto end_time = std::chrono::steady_clock::now();
    LOG(INFO) << "Pure Indexing creation in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                  end_time - start_start).count() << "[ms]";
    auto index = std::make_shared<HashIndex<ARROW_T, CTYPE>>(index_column, input_table->num_rows(), pool, out_umm_ptr);

    return index;
  };

};

template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
class LinearIndexKernel : public IndexKernel {
 public:
  LinearIndexKernel() : IndexKernel() {

  };

  using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<ARROW_T>::ArrayType;

  std::shared_ptr<BaseIndex> BuildIndex(arrow::MemoryPool *pool,
                                        std::shared_ptr<arrow::Table> &input_table,
                                        const int index_column) override {
    std::shared_ptr<arrow::Array> index_array;

    if (input_table->column(0)->num_chunks() > 1) {
      const arrow::Result<std::shared_ptr<arrow::Table>> &res = input_table->CombineChunks(pool);
      if (!res.status().ok()) {
        LOG(ERROR) << "Error occurred in combining chunks in table";
      }
      input_table = res.ValueOrDie();
    }

    index_array = cylon::util::GetChunkOrEmptyArray(input_table->column(index_column), 0);
    auto cast_index_array = std::static_pointer_cast<ARROW_ARRAY_TYPE>(index_array);
    auto
        index =
        std::make_shared<LinearIndex<ARROW_T, CTYPE>>(index_column, input_table->num_rows(), pool, cast_index_array);

    return index;
  }
};

/**
 * Hash Indexing Kernels
 * */
using BoolHashIndexKernel = HashIndexKernel<arrow::BooleanType>;
using UInt8HashIndexKernel = HashIndexKernel<arrow::UInt8Type>;
using Int8HashIndexKernel = HashIndexKernel<arrow::Int8Type>;
using UInt16HashIndexKernel = HashIndexKernel<arrow::UInt16Type>;
using Int16HashIndexKernel = HashIndexKernel<arrow::Int16Type>;
using UInt32HashIndexKernel = HashIndexKernel<arrow::UInt32Type>;
using Int32HashIndexKernel = HashIndexKernel<arrow::Int32Type>;
using UInt64HashIndexKernel = HashIndexKernel<arrow::UInt64Type>;
using Int64HashIndexKernel = HashIndexKernel<arrow::Int64Type>;
using HalfFloatHashIndexKernel = HashIndexKernel<arrow::HalfFloatType>;
using FloatHashIndexKernel = HashIndexKernel<arrow::FloatType>;
using DoubleHashIndexKernel = HashIndexKernel<arrow::DoubleType>;
using StringHashIndexKernel = HashIndexKernel<arrow::StringType, arrow::util::string_view>;
using BinaryHashIndexKernel = HashIndexKernel<arrow::BinaryType, arrow::util::string_view>;

/**
 * Range Indexing Kernel
 * */
using GenericRangeIndexKernel = RangeIndexKernel;

/*
 * Linear Indexing Kernel
 * **/

using BoolLinearIndexKernel = LinearIndexKernel<arrow::BooleanType>;
using UInt8LinearIndexKernel = LinearIndexKernel<arrow::UInt8Type>;
using Int8LinearIndexKernel = LinearIndexKernel<arrow::Int8Type>;
using UInt16LinearIndexKernel = LinearIndexKernel<arrow::UInt16Type>;
using Int16LinearIndexKernel = LinearIndexKernel<arrow::Int16Type>;
using UInt32LinearIndexKernel = LinearIndexKernel<arrow::UInt32Type>;
using Int32LinearIndexKernel = LinearIndexKernel<arrow::Int32Type>;
using UInt64LinearIndexKernel = LinearIndexKernel<arrow::UInt64Type>;
using Int64LinearIndexKernel = LinearIndexKernel<arrow::Int64Type>;
using HalfFloatLinearIndexKernel = LinearIndexKernel<arrow::HalfFloatType>;
using FloatLinearIndexKernel = LinearIndexKernel<arrow::FloatType>;
using DoubleLinearIndexKernel = LinearIndexKernel<arrow::DoubleType>;
using StringLinearIndexKernel = LinearIndexKernel<arrow::StringType, arrow::util::string_view>;
using BinaryLinearIndexKernel = LinearIndexKernel<arrow::BinaryType, arrow::util::string_view>;

std::unique_ptr<IndexKernel> CreateHashIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column);

std::unique_ptr<IndexKernel> CreateLinearIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column);

std::unique_ptr<IndexKernel> CreateIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column);

}

#endif //CYLON_SRC_CYLON_INDEXING_INDEX_H_
