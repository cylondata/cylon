
#ifndef CYLON_SRC_CYLON_INDEXING_INDEXER_H_
#define CYLON_SRC_CYLON_INDEXING_INDEXER_H_

#include "index.hpp"
#include "table.hpp"

namespace cylon {

//template<typename Base, typename T>
//inline bool instanceof(const T *) {
//  return std::is_base_of<Base, T>::value;
//}
//
//template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
//cylon::Status IsIndexValueUnique(const void *index_value, const std::shared_ptr<BaseIndex> &index, bool &is_unique) {
//  if (std::shared_ptr<cylon::RangeIndex> r = std::dynamic_pointer_cast<cylon::RangeIndex>(index)) {
//    LOG(INFO) << "Range Index detected";
//    is_unique = true;
//    return cylon::Status::OK();
//  }
//
//  LOG(INFO) << "Non-Range Index detected";
//
//  using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<ARROW_T>::ArrayType;
//  auto index_arr = index->GetIndexArray();
//  is_unique = true;
//  const CTYPE index_val = *static_cast<const CTYPE *>(index_value);
//  int64_t find_cout = 0;
//  auto reader0 = std::static_pointer_cast<ARROW_ARRAY_TYPE>(index_arr);
//  for (int64_t ix = 0; ix < reader0->length(); ix++) {
//    auto val = reader0->GetView(ix);
//    if (val == index_val) {
//      find_cout++;
//    }
//    if (find_cout > 1) {
//      is_unique = false;
//      break;
//    }
//  }
//  return cylon::Status::OK();
//}

//template<>
//cylon::Status IsIndexValueUnique<arrow::StringType, arrow::util::string_view>(const void *index_value,
//                                                                              const std::shared_ptr<BaseIndex> &index,
//                                                                              bool &is_unique) {
//  if (std::shared_ptr<cylon::RangeIndex> r = std::dynamic_pointer_cast<cylon::RangeIndex>(index)) {
//    is_unique = true;
//    return cylon::Status::OK();
//  }
//  using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<arrow::StringType>::ArrayType;
//  auto index_arr = index->GetIndexArray();
//  is_unique = true;
//  const std::string *sp = static_cast<const std::string *>(index_value);
//  arrow::util::string_view search_param_sv(*sp);
//  int64_t find_cout = 0;
//  auto reader0 = std::static_pointer_cast<ARROW_ARRAY_TYPE>(index_arr);
//  for (int64_t ix = 0; ix < reader0->length(); ix++) {
//    arrow::util::string_view val = reader0->GetView(ix);
//    if (search_param_sv == val) {
//      find_cout++;
//    }
//    if (find_cout > 1) {
//      is_unique = false;
//      break;
//    }
//  }
//  return cylon::Status::OK();
//}

//cylon::Status CheckIsIndexValueUnique(const void *index_value,
//                                      const std::shared_ptr<BaseIndex> &index,
//                                      bool &is_unique);

cylon::Status CheckIsIndexValueUnique(const std::shared_ptr<arrow::Scalar> &index_value,
									  const std::shared_ptr<BaseArrowIndex> &index,
									  bool &is_unique);

class BaseIndexer {

 public:
  explicit BaseIndexer() {

  }

  virtual Status loc(const void *start_index,
                     const void *end_index,
                     const int column_index,
                     const std::shared_ptr<Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const void *start_index,
                     const void *end_index,
                     const int start_column_index,
                     const int end_column_index,
                     const std::shared_ptr<Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const void *start_index,
                     const void *end_index,
                     const std::vector<int> &columns,
                     const std::shared_ptr<Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const void *indices,
                     const int column_index,
                     const std::shared_ptr<Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const void *indices,
                     const int start_column,
                     const int end_column,
                     const std::shared_ptr<Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const void *indices,
                     const std::vector<int> &columns,
                     const std::shared_ptr<Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::vector<void *> &indices,
                     const int column,
                     const std::shared_ptr<Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::vector<void *> &indices,
                     const int start_column_index,
                     const int end_column_index,
                     const std::shared_ptr<Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::vector<void *> &indices,
                     const std::vector<int> &columns,
                     const std::shared_ptr<Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual IndexingSchema GetIndexingSchema() = 0;

};

/**
 * Loc operations
 * */


//class LocIndexer : public BaseIndexer {
//
// public:
//  LocIndexer(IndexingSchema indexing_schema) : BaseIndexer(), indexing_schema_(indexing_schema) {
//
//  };
//
//  Status loc(const void *start_index,
//             const void *end_index,
//             const int column_index,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const void *start_index,
//             const void *end_index,
//             const int start_column_index,
//             const int end_column_index,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const void *start_index,
//             const void *end_index,
//             const std::vector<int> &columns,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const void *indices,
//             const int column_index,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const void *indices,
//             const int start_column,
//             const int end_column,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const void *indices,
//             const std::vector<int> &columns,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const std::vector<void *> &indices,
//             const int column,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const std::vector<void *> &indices,
//             const int start_column_index,
//             const int end_column_index,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const std::vector<void *> &indices,
//             const std::vector<int> &columns,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//
//  IndexingSchema GetIndexingSchema() override;
//
// private:
//  IndexingSchema indexing_schema_;
//
//};

//template<typename CTYPE>
//class TLocIndexer : public LocIndexer{
// public:
//  TLocIndexer(IndexingSchema indexing_schema) : LocIndexer(indexing_schema) {
//
//  }
//
//};

class ArrowBaseIndexer {

 public:
  explicit ArrowBaseIndexer() {

  }

  virtual Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
					 const std::shared_ptr<arrow::Scalar> &end_index,
					 const int column_index,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
					 const std::shared_ptr<arrow::Scalar> &end_index,
					 const int start_column_index,
					 const int end_column_index,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
					 const std::shared_ptr<arrow::Scalar> &end_index,
					 const std::vector<int> &columns,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::shared_ptr<arrow::Array> &indices,
					 const int column_index,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::shared_ptr<arrow::Array> &indices,
					 const int start_column,
					 const int end_column,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(const std::shared_ptr<arrow::Array> &indices,
					 const std::vector<int> &columns,
					 const std::shared_ptr<Table> &input_table,
					 std::shared_ptr<cylon::Table> &output) = 0;

  virtual IndexingSchema GetIndexingSchema() = 0;

};

class ArrowLocIndexer : public ArrowBaseIndexer {

 public:
  explicit ArrowLocIndexer(IndexingSchema indexing_schema) : ArrowBaseIndexer(), indexing_schema_(indexing_schema) {

  };

  Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
			 const std::shared_ptr<arrow::Scalar> &end_index,
			 const int column_index,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
			 const std::shared_ptr<arrow::Scalar> &end_index,
			 const int start_column_index,
			 const int end_column_index,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
			 const std::shared_ptr<arrow::Scalar> &end_index,
			 const std::vector<int> &columns,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Array> &indices,
			 const int column_index,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Array> &indices,
			 const int start_column,
			 const int end_column,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Array> &indices,
			 const std::vector<int> &columns,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  IndexingSchema GetIndexingSchema() override;

 private:
  IndexingSchema indexing_schema_;

};



/**
 * iLoc operations
 * */

//class ILocIndexer : public LocIndexer {
// public:
//  ILocIndexer(IndexingSchema indexing_schema) : LocIndexer(indexing_schema), indexing_schema_(indexing_schema) {
//
//  };
//
//  Status loc(const void *start_index,
//             const void *end_index,
//             const int column_index,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const void *start_index,
//             const void *end_index,
//             const int start_column_index,
//             const int end_column_index,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const void *start_index,
//             const void *end_index,
//             const std::vector<int> &columns,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const void *indices,
//             const int column_index,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const void *indices,
//             const int start_column,
//             const int end_column,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const void *indices,
//             const std::vector<int> &columns,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const std::vector<void *> &indices,
//             const int column,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const std::vector<void *> &indices,
//             const int start_column_index,
//             const int end_column_index,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//  Status loc(const std::vector<void *> &indices,
//             const std::vector<int> &columns,
//             const std::shared_ptr<Table> &input_table,
//             std::shared_ptr<cylon::Table> &output) override;
//
//  IndexingSchema GetIndexingSchema() override;
//
// private:
//  IndexingSchema indexing_schema_;
//};


class ArrowILocIndexer : public ArrowLocIndexer{
 public:
  ArrowILocIndexer(IndexingSchema indexing_schema);

  Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
			 const std::shared_ptr<arrow::Scalar> &end_index,
			 const int column_index,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
			 const std::shared_ptr<arrow::Scalar> &end_index,
			 const int start_column_index,
			 const int end_column_index,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Scalar> &start_index,
			 const std::shared_ptr<arrow::Scalar> &end_index,
			 const std::vector<int> &columns,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Array> &indices,
			 const int column_index,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Array> &indices,
			 const int start_column,
			 const int end_column,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  Status loc(const std::shared_ptr<arrow::Array> &indices,
			 const std::vector<int> &columns,
			 const std::shared_ptr<Table> &input_table,
			 std::shared_ptr<cylon::Table> &output) override;
  IndexingSchema GetIndexingSchema() override;

 private:
  IndexingSchema indexing_schema_;
};



}

#endif //CYLON_SRC_CYLON_INDEXING_INDEXER_H_
