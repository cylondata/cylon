
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
  //virtual void SetIndex(void *index_object) = 0;
  //virtual void *GetIndexSet() = 0;

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



//  void SetIndex(MMAP_TYPE map) override {
//    index_set_ = map;
//  }

//  const MMAP_TYPE &GetIndexSet() const  override  {
//    return index_set_;
//  };
 private:
  int col_ids_;
  int size_;
  MMAP_TYPE  map_;
 public:

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
    // TODO:: casting Index to BaseIndex
    //auto base_index = std::make_shared<BaseIndex>(index_column, input_table->num_rows());
    auto index = std::make_shared<Index<ARROW_T, CTYPE>>(index_column, input_table->num_rows(), out_umm_ptr);
    return index;
  };

};

using Int64HashIndexKernel = HashIndexKernel<arrow::Int64Type>;

std::unique_ptr<IndexKernel> CreateHashIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column);




//template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
//class HashIndexingKernel {
//  using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<ARROW_T>::ArrayType;
//  using MMAP_TYPE = typename std::unordered_multimap<CTYPE, int64_t>;
//
// public:
//  std::shared_ptr<BaseIndex> BuildIndex(const std::shared_ptr<arrow::Array> &left_idx_col);
//
//};
//template<class ARROW_T, typename CTYPE>
//std::shared_ptr<BaseIndex> HashIndexingKernel<ARROW_T,
//                                 CTYPE>::BuildIndex(const std::shared_ptr<arrow::Array> &left_idx_col
//) {
//  std::shared_ptr<Index<ARROW_T, CTYPE>> index;
//  MMAP_TYPE out_umm_ptr = MMAP_TYPE(left_idx_col->length());
//  auto reader0 = std::static_pointer_cast<ARROW_ARRAY_TYPE>(out_umm_ptr);
//  // TODO: add logic here for hash by value
//  for (int64_t i = 0; i < reader0->length(); i++) {
//    auto val = reader0->GetView(i);
//    out_umm_ptr.insert(std::make_pair(val, i));
//    out_umm_ptr.emplace(val);
//  }
//  index.get()->SetIndex(out_umm_ptr);
//  return index;
//}

}

#endif //CYLON_SRC_CYLON_INDEXING_INDEX_H_
