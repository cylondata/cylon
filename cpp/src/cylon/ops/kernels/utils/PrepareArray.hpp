#ifndef CYLON_SRC_CYLON_OPS_KERNELS_UTILS_PREPAREARRAY_HPP_
#define CYLON_SRC_CYLON_OPS_KERNELS_UTILS_PREPAREARRAY_HPP_
#include <status.hpp>
#include <ctx/cylon_context.hpp>
#include <arrow/table.h>
#include <util/arrow_utils.hpp>
#include <ctx/arrow_memory_pool_utils.hpp>

namespace cylon {
namespace kernel {
/**
 * creates an Arrow array based on col_idx, filtered by row_indices
 * @param ctx
 * @param table
 * @param col_idx
 * @param row_indices
 * @param array_vector
 * @return
 */
Status PrepareArray(std::shared_ptr<CylonContext> ctx,
                    const std::shared_ptr<arrow::Table> &table,
                    const int32_t col_idx,
                    const std::shared_ptr<std::vector<int64_t>> &row_indices,
                    arrow::ArrayVector &array_vector);
}
}
#endif //CYLON_SRC_CYLON_OPS_KERNELS_UTILS_PREPAREARRAY_HPP_
