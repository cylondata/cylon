

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
										 const std::shared_ptr<arrow::Array> &index_array);

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
	index = std::make_shared<ArrowHashIndex<ARROW_T, CTYPE>>(-1, index_values->length(), pool, index_values);
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
