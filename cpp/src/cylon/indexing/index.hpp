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

class BaseArrowIndex {

 public:

  BaseArrowIndex(int col_id, int size, std::shared_ptr<CylonContext> &ctx) : BaseArrowIndex(col_id,
																							size,
																							cylon::ToArrowPool(ctx)) {

  };

  BaseArrowIndex(int col_id, int size, arrow::MemoryPool *pool) : size_(size), col_id_(col_id), pool_(pool) {

  };

  virtual Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
								 std::vector<int64_t> &find_index) = 0;

  virtual Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param, int64_t &find_index) = 0;

  virtual Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
								 const std::shared_ptr<arrow::Table> &input,
								 std::vector<int64_t> &filter_location,
								 std::shared_ptr<arrow::Table> &output) = 0;

  virtual Status LocationByVector(const std::shared_ptr<arrow::Array> &search_param,
								  std::vector<int64_t> &filter_location) = 0;

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

/**
 *  HashIndex
 * */

template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
class ArrowHashIndex : public BaseArrowIndex {
 public:
  using MMAP_TYPE = typename std::unordered_multimap<CTYPE, int64_t>;
  using SCALAR_TYPE = typename arrow::TypeTraits<ARROW_T>::ScalarType;
  ArrowHashIndex(int col_ids, int size, arrow::MemoryPool *pool, std::shared_ptr<MMAP_TYPE> map)
	  : BaseArrowIndex(col_ids, size, pool) {
	map_ = map;
  };

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
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

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
						 std::vector<int64_t> &find_index) override {
	std::shared_ptr<SCALAR_TYPE> casted_value = std::static_pointer_cast<SCALAR_TYPE>(search_param);
	const CTYPE val = casted_value->value;
	auto ret = map_->equal_range(val);
	for (auto it = ret.first; it != ret.second; ++it) {
	  find_index.push_back(it->second);
	}
	return Status::OK();
  }

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param, int64_t &find_index) override {
	std::shared_ptr<SCALAR_TYPE> casted_value = std::static_pointer_cast<SCALAR_TYPE>(search_param);
	const CTYPE val = casted_value->value;
	auto ret = map_->find(val);
	if (ret != map_->end()) {
	  find_index = ret->second;
	  return Status::OK();
	}
	return Status(cylon::Code::IndexError);
  }

  Status LocationByVector(const std::shared_ptr<arrow::Array> &search_param,
						  std::vector<int64_t> &filter_location) override {
	cylon::Status status;
	for (int64_t ix = 0; ix < search_param->length(); ix++) {
	  auto index_val_sclr = search_param->GetScalar(ix).ValueOrDie();
	  status = LocationByValue(index_val_sclr, filter_location);
	  RETURN_CYLON_STATUS_IF_FAILED(status);
	}
	return Status::OK();
  }

  std::shared_ptr<arrow::Array> GetIndexAsArray() override {

	using ARROW_ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;
	using ARROW_BUILDER_T = typename arrow::TypeTraits<ARROW_T>::BuilderType;

	arrow::Status arrow_status;
	auto pool = GetPool();

	ARROW_BUILDER_T builder(pool);

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
	return BaseArrowIndex::GetColId();
  }
  int GetSize() const override {
	return BaseArrowIndex::GetSize();
  }
  arrow::MemoryPool *GetPool() const override {
	return BaseArrowIndex::GetPool();
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
class ArrowHashIndex<arrow::StringType, arrow::util::string_view> : public BaseArrowIndex {
 public:
  using MMAP_TYPE = typename std::unordered_multimap<arrow::util::string_view, int64_t>;
  using SCALAR_TYPE = typename arrow::TypeTraits<arrow::StringType>::ScalarType;
  ArrowHashIndex(int col_ids, int size, arrow::MemoryPool *pool, std::shared_ptr<MMAP_TYPE> map)
	  : BaseArrowIndex(col_ids, size, pool) {
	map_ = map;
  };

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
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

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
						 std::vector<int64_t> &find_index) override {
	LOG(INFO) << "Finding row ids for a given index";
	std::shared_ptr<SCALAR_TYPE> casted_value = std::static_pointer_cast<SCALAR_TYPE>(search_param);
	const std::string val = (casted_value->value->ToString());
	//const std::string *sp = static_cast<const std::string *>(search_param);
	//arrow::util::string_view search_param_sv(*sp);
	auto ret = map_->equal_range(val);
	for (auto it = ret.first; it != ret.second; ++it) {
	  find_index.push_back(it->second);
	}
	return Status::OK();
  };

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param, int64_t &find_index) override {
	LOG(INFO) << "Finding row id for a given index";
//	const std::string *sp = static_cast<const std::string *>(search_param);
//	arrow::util::string_view search_param_sv(*sp);
	std::shared_ptr<SCALAR_TYPE> casted_value = std::static_pointer_cast<SCALAR_TYPE>(search_param);
	const std::string val = (casted_value->value->ToString());
	auto ret = map_->find(val);
	if (ret != map_->end()) {
	  find_index = ret->second;
	  return Status::OK();
	}
	return Status(cylon::Code::IndexError);
  };

  Status LocationByVector(const std::shared_ptr<arrow::Array> &search_param,
						  std::vector<int64_t> &filter_location) override {
	cylon::Status status;
	for (int64_t ix = 0; ix < search_param->length(); ix++) {
	  std::vector<int64_t> filter_ix;
	  auto index_val_sclr = search_param->GetScalar(ix).ValueOrDie();

	  status = LocationByValue(index_val_sclr, filter_ix);
	  if (!status.is_ok()) {
		LOG(ERROR) << "Error in retrieving indices!";
		return status;
	  }

	  for (size_t iy = 0; iy < filter_ix.size(); iy++) {
		filter_location.push_back(filter_ix.at(iy));
	  }
	}

	return Status::OK();
  }

  std::shared_ptr<arrow::Array> GetIndexAsArray() override {

	arrow::Status arrow_status;
	auto pool = GetPool();

	arrow::StringBuilder builder(pool);

	std::shared_ptr<arrow::StringArray> index_array;

	std::vector<std::string> vec(GetSize(), "");

	for (const auto &x: *map_) {
	  vec[x.second] = x.first.to_string();
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
	return BaseArrowIndex::GetColId();
  }
  int GetSize() const override {
	return BaseArrowIndex::GetSize();
  }
  arrow::MemoryPool *GetPool() const override {
	return BaseArrowIndex::GetPool();
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


/*
 * Arrow HashIndex
 * **/

/**
 * End of HashIndex
 * */

class ArrowRangeIndex : public BaseArrowIndex {
 public:
  ArrowRangeIndex(int start, int size, int step, arrow::MemoryPool *pool);

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param, std::vector<int64_t> &find_index) override;
  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param, int64_t &find_index) override;
  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
						 const std::shared_ptr<arrow::Table> &input,
						 std::vector<int64_t> &filter_location,
						 std::shared_ptr<arrow::Table> &output) override;
  Status LocationByVector(const std::shared_ptr<arrow::Array> &search_param,
						  std::vector<int64_t> &filter_location) override;
  std::shared_ptr<arrow::Array> GetIndexAsArray() override;
  void SetIndexArray(std::shared_ptr<arrow::Array> &index_arr) override;
  std::shared_ptr<arrow::Array> GetIndexArray() override;
  int GetColId() const override;
  int GetSize() const override;
  IndexingSchema GetSchema() override;
  arrow::MemoryPool *GetPool() const override;
  bool IsUnique() override;

  int GetStart() const;
  int GetAnEnd() const;
  int GetStep() const;

 private:
  int start_ = 0;
  int end_ = 0;
  int step_ = 1;
  std::shared_ptr<arrow::Array> index_arr_ = nullptr;
};

class ArrowLinearIndex : public BaseArrowIndex {
 public:
  ArrowLinearIndex(int col_id, int size, std::shared_ptr<CylonContext> &ctx);
  ArrowLinearIndex(int col_id, int size, arrow::MemoryPool *pool);
  ArrowLinearIndex(int col_id, int size, arrow::MemoryPool *pool, std::shared_ptr<arrow::Array> &index_array)
	  : BaseArrowIndex(col_id, size, pool), index_array_(index_array) {
  }

  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param, std::vector<int64_t> &find_index) override;
  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param, int64_t &find_index) override;
  Status LocationByValue(const std::shared_ptr<arrow::Scalar> &search_param,
						 const std::shared_ptr<arrow::Table> &input,
						 std::vector<int64_t> &filter_location,
						 std::shared_ptr<arrow::Table> &output) override;
  Status LocationByVector(const std::shared_ptr<arrow::Array> &search_param,
						  std::vector<int64_t> &filter_location) override;
  std::shared_ptr<arrow::Array> GetIndexAsArray() override;
  void SetIndexArray(std::shared_ptr<arrow::Array> &index_arr) override;
  std::shared_ptr<arrow::Array> GetIndexArray() override;
  int GetColId() const override;
  int GetSize() const override;
  IndexingSchema GetSchema() override;
  arrow::MemoryPool *GetPool() const override;
  bool IsUnique() override;

 private:
  std::shared_ptr<arrow::Array> index_array_;

};

class ArrowIndexKernel {
 public:
  explicit ArrowIndexKernel() {

  }
  virtual std::shared_ptr<BaseArrowIndex> BuildIndex(arrow::MemoryPool *pool,
													 std::shared_ptr<arrow::Table> &input_table,
													 const int index_column) = 0;
};

class ArrowRangeIndexKernel : public ArrowIndexKernel {
 public:
  ArrowRangeIndexKernel();

  std::shared_ptr<BaseArrowIndex> BuildIndex(arrow::MemoryPool *pool,
											 std::shared_ptr<arrow::Table> &input_table,
											 const int index_column) override;

};

template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
class ArrowHashIndexKernel : public ArrowIndexKernel {

 public:
  explicit ArrowHashIndexKernel() : ArrowIndexKernel() {}
  using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using MMAP_TYPE = typename std::unordered_multimap<CTYPE, int64_t>;

  std::shared_ptr<BaseArrowIndex> BuildIndex(arrow::MemoryPool *pool,
											 std::shared_ptr<arrow::Table> &input_table,
											 const int index_column) override {

	const std::shared_ptr<arrow::Array>
		&idx_column = cylon::util::GetChunkOrEmptyArray(input_table->column(index_column), 0);
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
	auto index =
		std::make_shared<ArrowHashIndex<ARROW_T, CTYPE>>(index_column, input_table->num_rows(), pool, out_umm_ptr);

	return index;
  };

};

class LinearArrowIndexKernel : public ArrowIndexKernel {

 public:
  LinearArrowIndexKernel();

  std::shared_ptr<BaseArrowIndex> BuildIndex(arrow::MemoryPool *pool,
											 std::shared_ptr<arrow::Table> &input_table,
											 const int index_column) override;

};


/**
 * Hash Indexing Kernels
 * */

/**
 * Arrow Hash Index Kernels
 *
 * */

using BoolArrowHashIndexKernel = ArrowHashIndexKernel<arrow::BooleanType>;
using UInt8ArrowHashIndexKernel = ArrowHashIndexKernel<arrow::UInt8Type>;
using Int8ArrowHashIndexKernel = ArrowHashIndexKernel<arrow::Int8Type>;
using UInt16ArrowHashIndexKernel = ArrowHashIndexKernel<arrow::UInt16Type>;
using Int16ArrowHashIndexKernel = ArrowHashIndexKernel<arrow::Int16Type>;
using UInt32ArrowHashIndexKernel = ArrowHashIndexKernel<arrow::UInt32Type>;
using Int32ArrowHashIndexKernel = ArrowHashIndexKernel<arrow::Int32Type>;
using UInt64ArrowHashIndexKernel = ArrowHashIndexKernel<arrow::UInt64Type>;
using Int64ArrowHashIndexKernel = ArrowHashIndexKernel<arrow::Int64Type>;
using HalfFloatArrowHashIndexKernel = ArrowHashIndexKernel<arrow::HalfFloatType>;
using FloatArrowHashIndexKernel = ArrowHashIndexKernel<arrow::FloatType>;
using DoubleArrowHashIndexKernel = ArrowHashIndexKernel<arrow::DoubleType>;
using StringArrowHashIndexKernel = ArrowHashIndexKernel<arrow::StringType, arrow::util::string_view>;
using BinaryArrowHashIndexKernel = ArrowHashIndexKernel<arrow::BinaryType, arrow::util::string_view>;


/**
 * Range Indexing Kernel
 * */

std::unique_ptr<ArrowIndexKernel> CreateArrowHashIndexKernel(std::shared_ptr<arrow::Table> input_table,
															 int index_column);

std::unique_ptr<ArrowIndexKernel> CreateArrowIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column);

}

#endif //CYLON_SRC_CYLON_INDEXING_INDEX_H_
