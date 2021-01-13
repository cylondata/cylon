#ifndef CYLON_SRC_CYLON_INDEXING_INDEX_H_
#define CYLON_SRC_CYLON_INDEXING_INDEX_H_

#include "index.hpp"

#include "status.hpp"
#include "ctx/cylon_context.hpp"
#include "ctx/arrow_memory_pool_utils.hpp"
#include "util/macros.hpp"

#include <arrow/table.h>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/compute/kernel.h>
#include <arrow/arrow_comparator.hpp>

namespace cylon {

class BaseIndex {

 public:
  explicit BaseIndex(int col_id, int size, arrow::MemoryPool *pool) {
    col_id_ = col_id;
    size_ = size;
    pool_ = pool;
  };

  // TODO: virtual destructor
  //virtual void SetIndex(void *index_object) = 0;
  //virtual void *GetIndexSet() = 0;

  virtual Status Find(void *search_param,
                      std::shared_ptr<arrow::Table> &input,
                      std::shared_ptr<arrow::Table> &output) = 0;

  virtual std::shared_ptr<arrow::Array> GetIndex() = 0;

  int GetColId() const {
    return col_id_;
  }
  int GetSize() const {
    return size_;
  }

  arrow::MemoryPool *GetPool() const {
    return pool_;
  }

 private:
  int size_;
  int col_id_;
  arrow::MemoryPool *pool_;
};

template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
class Index : public BaseIndex {
 public:
  using MMAP_TYPE = typename std::unordered_multimap<CTYPE, int64_t>;
  explicit Index(int col_ids, int size, arrow::MemoryPool *pool, std::shared_ptr<MMAP_TYPE> map)
      : BaseIndex(col_ids, size, pool) {
    map_ = map;
  };

  Status Find(void *search_param,
              std::shared_ptr<arrow::Table> &input,
              std::shared_ptr<arrow::Table> &output) override {

    arrow::Status arrow_status;
    std::shared_ptr<arrow::Array> out_idx;
    CTYPE val = *static_cast<CTYPE *>(search_param);
    auto ret = map_->equal_range(val);
    arrow::compute::ExecContext fn_ctx(GetPool());
    arrow::Int64Builder idx_builder(GetPool());
    const arrow::Datum input_table(input);
    std::vector<int64_t> filter_vals;
    for (auto it = ret.first; it != ret.second; ++it) {
      filter_vals.push_back(it->second);
    }
    idx_builder.AppendValues(filter_vals);
    arrow_status = idx_builder.Finish(&out_idx);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(arrow_status);
    const arrow::Datum filter_indices(out_idx);
    arrow::Result<arrow::Datum>
        result = arrow::compute::Take(input_table, filter_indices, arrow::compute::TakeOptions::Defaults(), &fn_ctx);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(result.status());
    output = result.ValueOrDie().table();
    return Status::OK();
  }

  std::shared_ptr<arrow::Array> GetIndex() override {

    using ARROW_ARRAY_T = typename arrow::TypeTraits<ARROW_T>::ArrayType;
    using ARROW_BUILDER_T = typename arrow::TypeTraits<ARROW_T>::BuilderType;

    arrow::Status arrow_status;

    // TODO :: add ctx to Index
    auto pool = GetPool();
    ARROW_BUILDER_T builder(pool);

    std::shared_ptr<ARROW_ARRAY_T> index_array;

    std::vector<CTYPE> vec(GetSize(), 1);
    LOG(INFO) << "Get Index :: " << GetSize();

    for (const auto &x: *map_) {
      std::cout << x.first << ":" << x.second << std::endl;
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

 private:
  int col_ids_;
  int size_;
  std::shared_ptr<MMAP_TYPE> map_;

};

class IndexKernel {
 public:
  explicit IndexKernel() {

  }
  virtual std::shared_ptr<BaseIndex> BuildIndex(arrow::MemoryPool *pool,
                                                std::shared_ptr<arrow::Table> &input_table,
                                                const int index_column) = 0;
};

template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
class HashIndexKernel : public IndexKernel {

 public:
  explicit HashIndexKernel() : IndexKernel() {}
  // TODO: move like the following way
  // using CTYPE = typename ARROW_T::c_type;
  using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using MMAP_TYPE = typename std::unordered_multimap<CTYPE, int64_t>;

  std::shared_ptr<BaseIndex> BuildIndex(arrow::MemoryPool *pool,
                                        std::shared_ptr<arrow::Table> &input_table,
                                        const int index_column) override {

    const std::shared_ptr<arrow::Array> &idx_column = input_table->column(index_column)->chunk(0);
    std::shared_ptr<MMAP_TYPE> out_umm_ptr = std::make_shared<MMAP_TYPE>(idx_column->length());
    auto reader0 = std::static_pointer_cast<ARROW_ARRAY_TYPE>(idx_column);
    for (int64_t i = 0; i < reader0->length(); i++) {
      auto val = reader0->GetView(i);
      out_umm_ptr->insert(std::make_pair(val, i));
    }
    auto index = std::make_shared<Index<ARROW_T, CTYPE>>(index_column, input_table->num_rows(), pool, out_umm_ptr);

    return index;
  };

};

using Int64HashIndexKernel = HashIndexKernel<arrow::Int64Type>;

std::unique_ptr<IndexKernel> CreateHashIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column);

}

#endif //CYLON_SRC_CYLON_INDEXING_INDEX_H_
