#include "union_op.hpp"

namespace cylon {

UnionOp::UnionOp(std::shared_ptr<cylon::CylonContext> ctx,
                 std::shared_ptr<arrow::Schema> schema,
                 int id,
                 std::function<int(int)> router,
                 std::shared_ptr<ResultsCallback> callback,
                 std::shared_ptr<UnionOpConfig> config) : Op(ctx, schema, id, router, callback), config(config) {
  this->union_kernel = new cylon::kernel::Union(ctx, schema, config->GetExpectedRows());
}

bool UnionOp::Execute(int tag, std::shared_ptr<Table> table) {
  this->union_kernel->InsertTable(table);
  return true;
}

void UnionOp::Finalize() {
  std::shared_ptr<cylon::Table> final_result;
  this->union_kernel->Finalize(final_result);

  this->InsertToAllChildren(0, final_result);
}

int64_t UnionOpConfig::GetExpectedRows() const {
  return expected_rows;
}

void UnionOpConfig::SetExpectedRows(int64_t expected_final_rows) {
  UnionOpConfig::expected_rows = expected_final_rows;
}
}