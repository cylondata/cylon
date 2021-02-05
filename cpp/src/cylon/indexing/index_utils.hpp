

#ifndef CYLON_SRC_CYLON_INDEXING_BUILDER_H_
#define CYLON_SRC_CYLON_INDEXING_BUILDER_H_

#include "index.hpp"
#include "status.hpp"
#include "table.hpp"

namespace cylon {

class IndexUtil {

 public:

  static Status BuildIndexFromArray(const IndexingSchema schema,
                                    const std::shared_ptr<Table> &input,
                                    const std::shared_ptr<arrow::Array> &index_array,
                                    std::shared_ptr<Table> &output);

  static Status BuildIndex(const IndexingSchema schema,
                           const std::shared_ptr<Table> &input,
                           const int index_column,
                           const bool drop,
                           std::shared_ptr<Table> &output);

  static Status BuildIndex(const IndexingSchema schema,
                           const std::shared_ptr<Table> &input,
                           const int index_column,
                           std::shared_ptr<cylon::BaseIndex> &index);

  static Status BuildHashIndex(const std::shared_ptr<Table> &input,
                               const int index_column,
                               std::shared_ptr<cylon::BaseIndex> &index);

  static Status BuildLinearIndex(const std::shared_ptr<Table> &input,
                                 const int index_column,
                                 std::shared_ptr<cylon::BaseIndex> &index);

  static Status BuildRangeIndex(const std::shared_ptr<Table> &input, std::shared_ptr<cylon::BaseIndex> &index);


  template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
  static Status BuildHashIndexFromArrowArray(std::shared_ptr<arrow::Array> &index_values,
                                             arrow::MemoryPool *pool,
                                             std::shared_ptr<cylon::BaseIndex> &index) {
    using SCALAR_T = typename arrow::TypeTraits<ARROW_T>::ScalarType;
    using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<ARROW_T>::ArrayType;
    using MMAP_TYPE = typename std::unordered_multimap<CTYPE, int64_t>;
    Status s;
    std::shared_ptr<MMAP_TYPE> out_umm_ptr = std::make_shared<MMAP_TYPE>(index_values->length());
    std::shared_ptr<SCALAR_T> scalar_val;
    auto reader0 = std::static_pointer_cast<ARROW_ARRAY_TYPE>(index_values);
    for (int64_t i = reader0->length() - 1; i >= 0; --i) {
      auto val = reader0->GetView(i);
      out_umm_ptr->emplace(val, i);
    }
    index = std::make_shared<HashIndex<ARROW_T, CTYPE>>(0, index_values->length(), pool, out_umm_ptr);
    index->SetIndexArray(index_values);
    return Status::OK();
  }

  template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
  static Status BuildLinearIndexFromArrowArray(std::shared_ptr<arrow::Array> &index_values,
                                               arrow::MemoryPool *pool,
                                               std::shared_ptr<cylon::BaseIndex> &index) {
    using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<ARROW_T>::ArrayType;
    auto cast_index_array = std::static_pointer_cast<ARROW_ARRAY_TYPE>(index_values);
    index = std::make_shared<LinearIndex<ARROW_T, CTYPE>>(0, index_values->length(), pool, cast_index_array);
    return Status::OK();
  }

  static Status BuildRangeIndexFromArray(std::shared_ptr<arrow::Array> &index_values,
                                         arrow::MemoryPool *pool,
                                         std::shared_ptr<cylon::BaseIndex> &index) {

    index = std::make_shared<RangeIndex>(0, index_values->length(), 1, pool);
    index->SetIndexArray(index_values);
    return Status::OK();
  }

  static Status BuildHashIndexFromArray(std::shared_ptr<arrow::Array> &index_values,
                                        arrow::MemoryPool *pool,
                                        std::shared_ptr<cylon::BaseIndex> &index);

  static Status BuildLinearIndexFromArray(std::shared_ptr<arrow::Array> &index_values,
                                          arrow::MemoryPool *pool,
                                          std::shared_ptr<cylon::BaseIndex> &index);

  template<typename CTYPE, typename ARROW_T=typename arrow::CTypeTraits<CTYPE>::ArrowType>
  static Status BuildHashIndexFromVector(std::vector<CTYPE> &index_values,
                                         arrow::MemoryPool *pool,
                                         std::shared_ptr<cylon::BaseIndex> &index) {
    using MMAP_TYPE = typename std::unordered_multimap<CTYPE, int64_t>;
    std::shared_ptr<MMAP_TYPE> map = std::make_shared<MMAP_TYPE>(index_values.size());
    LOG(INFO) << "Start building index";
    for (int64_t i = index_values.size() - 1; i >= 0; i--) {
      map->template emplace(index_values.at(i), i);
    }
    LOG(INFO) << "Finished building index";
    index = std::make_shared<HashIndex<ARROW_T, CTYPE>>(0, index_values.size(), pool, map);
    return Status::OK();
  }

};
}

#endif //CYLON_SRC_CYLON_INDEXING_BUILDER_H_
