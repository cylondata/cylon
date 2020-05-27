#ifndef TWISTERX_SRC_TWISTERX_ARROW_ARROW_COMPARATOR_H_
#define TWISTERX_SRC_TWISTERX_ARROW_ARROW_COMPARATOR_H_

#include <arrow/api.h>
#include "../status.hpp"
namespace twisterx {
class ArrowComparator {
 public:
  virtual int compare(std::shared_ptr<arrow::Array> array1,
                      int64_t index1,
                      std::shared_ptr<arrow::Array> array2,
                      int64_t index2) = 0;
};

ArrowComparator *GetComparator(const std::shared_ptr<arrow::DataType> &type);

/**
 * This Util class can be used to compare the equality  of rows of two tables. This class
 * expect Tables to have just a single chunk in all its columns
 *
 * todo resolve single chunk expectation
 */
class TableRowComparator {
 private:
  std::vector<std::shared_ptr<ArrowComparator>> comparators;
 public:
  TableRowComparator(std::vector<std::shared_ptr<arrow::Field>> vector);
  int compare(const std::shared_ptr<arrow::Table> &table1,
              int64_t index1,
              const std::shared_ptr<arrow::Table> &table2,
              int64_t index2);
};

}

#endif //TWISTERX_SRC_TWISTERX_ARROW_ARROW_COMPARATOR_H_
