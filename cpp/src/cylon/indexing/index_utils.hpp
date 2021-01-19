

#ifndef CYLON_SRC_CYLON_INDEXING_BUILDER_H_
#define CYLON_SRC_CYLON_INDEXING_BUILDER_H_

#include "index.hpp"
#include "status.hpp"
#include "table.hpp"

namespace cylon {

class IndexUtil {

 public:
  static Status Build(std::shared_ptr<cylon::BaseIndex> &index,
                      std::shared_ptr<cylon::Table> &input,
                      int index_column);

  static Status Find(std::shared_ptr<cylon::BaseIndex> &index,
                     std::shared_ptr<cylon::Table> &find_table,
                     void *value,
                     int index_column,
                     std::shared_ptr<cylon::Table> &out);

  template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
  static Status BuildIndexFromArrowArray(std::shared_ptr<arrow::Array> &index_values,
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
    index = std::make_shared<Index<ARROW_T, CTYPE>>(0, index_values->length(), pool, out_umm_ptr);
    return Status::OK();
  }

  static Status BuildFromVector(std::shared_ptr<arrow::Array> &index_values,
                                arrow::MemoryPool *pool,
                                std::shared_ptr<cylon::BaseIndex> &index);

};
}

#endif //CYLON_SRC_CYLON_INDEXING_BUILDER_H_
