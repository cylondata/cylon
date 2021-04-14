

#ifndef CYLON_SRC_CYLON_INDEXING_BUILDER_H_
#define CYLON_SRC_CYLON_INDEXING_BUILDER_H_

#include "index.hpp"
#include "status.hpp"
#include "table.hpp"

namespace cylon {

class IndexUtil {

 public:

  static Status BuildArrowIndexFromArray(const IndexingType schema,
									const std::shared_ptr<Table> &input,
									const std::shared_ptr<arrow::Array> &index_array,
									std::shared_ptr<Table> &output);

  static Status BuildArrowIndex(IndexingType schema,
								const std::shared_ptr<Table> &input,
								int index_column,
								bool drop,
								std::shared_ptr<Table> &output);

  static Status BuildArrowIndex(const IndexingType schema,
								const std::shared_ptr<Table> &input,
								const int index_column,
								std::shared_ptr<cylon::BaseArrowIndex> &index);

  static Status BuildArrowHashIndex(const std::shared_ptr<Table> &input,
									const int index_column,
									std::shared_ptr<cylon::BaseArrowIndex> &index);

  static Status BuildArrowLinearIndex(const std::shared_ptr<Table> &input,
									  const int index_column,
									  std::shared_ptr<cylon::BaseArrowIndex> &index);

  static Status BuildArrowRangeIndex(const std::shared_ptr<Table> &input,
									 std::shared_ptr<cylon::BaseArrowIndex> &index);

  template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
  static Status BuildArrowHashIndexFromArrowArray(std::shared_ptr<arrow::Array> &index_values,
											 arrow::MemoryPool *pool,
											 std::shared_ptr<cylon::BaseArrowIndex> &index) {
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
	index = std::make_shared<ArrowHashIndex<ARROW_T, CTYPE>>(0, index_values->length(), pool, out_umm_ptr);
	index->SetIndexArray(index_values);
	return Status::OK();
  }

  static Status BuildArrowLinearIndexFromArrowArray(std::shared_ptr<arrow::Array> &index_values,
													arrow::MemoryPool *pool,
													std::shared_ptr<cylon::BaseArrowIndex> &index) {
	index = std::make_shared<ArrowLinearIndex>(0, index_values->length(), pool, index_values);
	return Status::OK();
  }

  static Status BuildArrowRangeIndexFromArray(int64_t size,
											  arrow::MemoryPool *pool,
											  std::shared_ptr<cylon::BaseArrowIndex> &index);

  static Status BuildArrowHashIndexFromArray(std::shared_ptr<arrow::Array> &index_values,
										arrow::MemoryPool *pool,
										std::shared_ptr<cylon::BaseArrowIndex> &index);

};
}

#endif //CYLON_SRC_CYLON_INDEXING_BUILDER_H_
