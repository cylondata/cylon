#include "PrepareArray.hpp"
#include <glog/logging.h>

namespace cylon {
namespace kernel {
Status PrepareArray(std::shared_ptr<CylonContext> ctx,
                    const std::shared_ptr<arrow::Table> &table,
                    const int32_t col_idx,
                    const std::shared_ptr<std::vector<int64_t>> &row_indices,
                    arrow::ArrayVector &array_vector) {
  std::shared_ptr<arrow::Array> destination_col_array;
  arrow::Status ar_status = cylon::util::copy_array_by_indices(row_indices,
                                                               table->column(col_idx)->chunk(0),
                                                               &destination_col_array, cylon::ToArrowPool(&*ctx));
  if (ar_status != arrow::Status::OK()) {
    LOG(FATAL) << "Failed while copying a column to the final table from tables."
               << ar_status.ToString();
    return Status(static_cast<int>(ar_status.code()), ar_status.message());
  }
  array_vector.push_back(destination_col_array);
  return Status::OK();
}
}
}