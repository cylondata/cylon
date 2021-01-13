

#include "index_utils.hpp"

cylon::Status cylon::IndexUtil::Build(std::shared_ptr<cylon::BaseIndex> &index,
                                      std::shared_ptr<cylon::Table> &input,
                                      int index_column,
                                      bool drop_index,
                                      std::shared_ptr<cylon::Table> &output) {

  std::shared_ptr<arrow::Table> arrow_out;

  auto table_ = input->get_table();
  auto ctx = input->GetContext();

  if (table_->column(0)->num_chunks() > 1) {
    const arrow::Result<std::shared_ptr<arrow::Table>> &res = table_->CombineChunks(cylon::ToArrowPool(ctx));
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(res.status())
    table_ = res.ValueOrDie();
  }

  auto pool = cylon::ToArrowPool(ctx);

  std::shared_ptr<cylon::IndexKernel> kernel = CreateHashIndexKernel(table_, index_column);
  std::shared_ptr<cylon::BaseIndex> bi = kernel->BuildIndex(pool, table_, index_column, drop_index, arrow_out);
  cylon::Table::FromArrowTable(ctx, arrow_out, output);
  index = std::move(bi);
  return cylon::Status::OK();
}

cylon::Status cylon::IndexUtil::Find(std::shared_ptr<cylon::BaseIndex> &index,
                                     std::shared_ptr<cylon::Table> &find_table,
                                     void *value,
                                     int index_column,
                                     std::shared_ptr<cylon::Table> &out) {
  std::shared_ptr<arrow::Table> ar_out;
  auto table_ = find_table->get_table();
  auto ctx = find_table->GetContext();
  if (index != nullptr && index->GetColId() == index_column) {
    index->Find(value, table_, ar_out);
    cylon::Table::FromArrowTable(ctx, ar_out, out);
  } else {
    LOG(ERROR) << "Index column doesn't match the provided column";
  }
  return cylon::Status::OK();
}
