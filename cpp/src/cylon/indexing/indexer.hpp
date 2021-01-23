
#ifndef CYLON_SRC_CYLON_INDEXING_INDEXER_H_
#define CYLON_SRC_CYLON_INDEXING_INDEXER_H_

#include "index.hpp"
#include "table.hpp"

namespace cylon {

template<typename Base, typename T>
inline bool instanceof(const T *) {
  return std::is_base_of<Base, T>::value;
}

template<class ARROW_T, typename CTYPE = typename ARROW_T::c_type>
cylon::Status IsIndexValueUnique(void *index_value, std::shared_ptr<cylon::BaseIndex> &index, bool &is_unique) {
  using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<ARROW_T>::ArrayType;
  auto index_arr = index->GetIndexArray();
  is_unique = true;
  CTYPE index_val = *static_cast<CTYPE *>(index_value);
  int64_t find_cout = 0;
  auto reader0 = std::static_pointer_cast<ARROW_ARRAY_TYPE>(index_arr);
  for (int64_t ix = 0; ix < reader0->length(); ix++) {
    auto val = reader0->GetView(ix);
    if (val == index_val) {
      find_cout++;
    }
    if (find_cout > 1) {
      is_unique = false;
      break;
    }
  }
  std::cout << index_val << ":: Find count : " << find_cout << "," << is_unique << std::endl;
  return cylon::Status::OK();
}

template<>
cylon::Status IsIndexValueUnique<arrow::StringType, arrow::util::string_view>(void *index_value,
                                                                              std::shared_ptr<cylon::BaseIndex> &index,
                                                                              bool &is_unique) {
  using ARROW_ARRAY_TYPE = typename arrow::TypeTraits<arrow::StringType>::ArrayType;
  auto index_arr = index->GetIndexArray();
  is_unique = true;
  std::string *sp = static_cast<std::string *>(index_value);
  arrow::util::string_view search_param_sv(*sp);
  int64_t find_cout = 0;
  auto reader0 = std::static_pointer_cast<ARROW_ARRAY_TYPE>(index_arr);
  for (int64_t ix = 0; ix < reader0->length(); ix++) {
    arrow::util::string_view val = reader0->GetView(ix);
    if (search_param_sv == val) {
      find_cout++;
    }
    if (find_cout > 1) {
      is_unique = false;
      break;
    }
  }
  std::cout << search_param_sv << ":: Find count : " << find_cout << "," << is_unique << std::endl;
  return cylon::Status::OK();
}

cylon::Status CheckIsIndexValueUnique(void *index_value,
                                      std::shared_ptr<cylon::BaseIndex> &index,
                                      bool &is_unique);

class BaseIndexer {

 public:
  explicit BaseIndexer() {

  }

  virtual Status loc(void *start_index,
                     void *end_index,
                     int column_index,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(void *start_index,
                     void *end_index,
                     int start_column_index,
                     int end_column_index,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(void *start_index,
                     void *end_index,
                     std::vector<int> &columns,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(void *indices,
                     int column_index,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(void *indices,
                     int start_column,
                     int end_column,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(void *indices,
                     std::vector<int> &columns,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(std::vector<void *> &indices,
                     int column,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(std::vector<void *> &indices,
                     int start_column_index,
                     int end_column_index,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual Status loc(std::vector<void *> &indices,
                     std::vector<int> &columns,
                     std::shared_ptr<cylon::Table> &input_table,
                     std::shared_ptr<cylon::Table> &output) = 0;

  virtual IndexingSchema GetIndexingSchema() = 0;

};

/**
 * Loc operations
 * */


class LocIndexer : public BaseIndexer {

 public:
  LocIndexer(IndexingSchema indexing_schema) : BaseIndexer(), indexing_schema_(indexing_schema) {

  };

  Status loc(void *start_index,
             void *end_index,
             int column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *start_index,
             void *end_index,
             int start_column_index,
             int end_column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *start_index,
             void *end_index,
             std::vector<int> &columns,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *indices,
             int column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *indices,
             int start_column,
             int end_column,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *indices,
             std::vector<int> &columns,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(std::vector<void *> &indices,
             int column,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(std::vector<void *> &indices,
             int start_column_index,
             int end_column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(std::vector<void *> &indices,
             std::vector<int> &columns,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;

  IndexingSchema GetIndexingSchema() override;

 private:
  IndexingSchema indexing_schema_;

};

/**
 * iLoc operations
 * */

class ILocIndexer : public BaseIndexer {
 public:
  ILocIndexer(IndexingSchema indexing_schema) : BaseIndexer(), indexing_schema_(indexing_schema) {

  };

  Status loc(void *start_index,
             void *end_index,
             int column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *start_index,
             void *end_index,
             int start_column_index,
             int end_column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *start_index,
             void *end_index,
             std::vector<int> &columns,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *indices,
             int column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *indices,
             int start_column,
             int end_column,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(void *indices,
             std::vector<int> &columns,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(std::vector<void *> &indices,
             int column,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(std::vector<void *> &indices,
             int start_column_index,
             int end_column_index,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;
  Status loc(std::vector<void *> &indices,
             std::vector<int> &columns,
             std::shared_ptr<cylon::Table> &input_table,
             std::shared_ptr<cylon::Table> &output) override;

  IndexingSchema GetIndexingSchema() override;

 private:
  IndexingSchema indexing_schema_;
};

}

#endif //CYLON_SRC_CYLON_INDEXING_INDEXER_H_
