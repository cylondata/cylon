
#ifndef CYLON_SRC_CYLON_INDEXING_INDEX_H_
#define CYLON_SRC_CYLON_INDEXING_INDEX_H_

#include <arrow/api.h>
#include <arrow/compute/kernel.h>
#include <arrow/arrow_comparator.hpp>

namespace cylon {

class BaseIndex {

 public:
  explicit BaseIndex(int col_id, int size) {
    col_id_ = col_id;
    size_ = size;
  };

  // TODO: virtual destructor
  //virtual void SetIndex(void *index_object) = 0;
  //virtual void *GetIndexSet() = 0;

  virtual void Find(void* search_param, std::shared_ptr<arrow::Table> &output) = 0;

 private:
  int col_id_;
  int size_;
};

template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
class Index : public BaseIndex {
 public:
  using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using MMAP_TYPE = typename std::unordered_multimap<CTYPE, int64_t>;
  explicit Index(int col_ids, int size, MMAP_TYPE map) : BaseIndex(col_ids, size) {
    map_ = map;
  };
  // TODO: add return type for Find func to handle error.
  void Find(void *search_param, std::shared_ptr<arrow::Table> &output) override {
    CTYPE val = *static_cast<CTYPE*>(search_param);
    std::cout << "Search Param : " << val << std::endl;
  }

 private:
  int col_ids_;
  int size_;
  MMAP_TYPE  map_;

};

class IndexKernel {
 public:
  explicit IndexKernel() {

  }
  virtual std::shared_ptr<BaseIndex> BuildIndex(std::shared_ptr<arrow::Table> &input_table,
                                                const int index_column,
                                                bool index_drop,
                                                std::shared_ptr<arrow::Table> &output_table) = 0;
};

template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
class HashIndexKernel : public IndexKernel {

 public:
  explicit HashIndexKernel() : IndexKernel() {}
  // TODO: move like the following way
  // using CTYPE = typename ARROW_T::c_type;
  using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  using MMAP_TYPE = typename std::unordered_multimap<CTYPE, int64_t>;

  std::shared_ptr<BaseIndex> BuildIndex(std::shared_ptr<arrow::Table> &input_table,
                                        const int index_column,
                                        bool index_drop,
                                        std::shared_ptr<arrow::Table> &output_table) override {

    const std::shared_ptr<arrow::Array> &idx_column = input_table->column(index_column)->chunk(0);
    MMAP_TYPE out_umm_ptr = MMAP_TYPE(idx_column->length());
    auto reader0 = std::static_pointer_cast<ARROW_ARRAY_TYPE>(idx_column);
    for (int64_t i = 0; i < reader0->length(); i++) {
      auto val = reader0->GetView(i);
      out_umm_ptr.insert(std::make_pair(val, i));
      out_umm_ptr.emplace(std::make_pair(val, i));
    }
    auto index = std::make_shared<Index<ARROW_T, CTYPE>>(index_column, input_table->num_rows(), out_umm_ptr);
    return index;
  };

};

using Int64HashIndexKernel = HashIndexKernel<arrow::Int64Type>;

std::unique_ptr<IndexKernel> CreateHashIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column);

}

#endif //CYLON_SRC_CYLON_INDEXING_INDEX_H_
