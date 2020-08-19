#include "test_header.hpp"
#include "test_utils.hpp"

using namespace cylon;

TEST_CASE("create table from columns testing", "[columns]") {
  cylon::Status status;
  const int size = 12;

  std::shared_ptr<std::vector<int32_t>> col0 = std::make_shared<std::vector<int32_t>>();
  std::shared_ptr<std::vector<double_t>> col1 = std::make_shared<std::vector<double_t>>();

  for (int i = 0; i < size; i++) {
    col0->push_back(i);
    col1->push_back((double_t) i + 10.0);
  }

  SECTION("testing inner joins") {
    auto cy_col0 =
        cylon::VectorColumn<int32_t>::Make("col0", cylon::DataType::Make(cylon::Type::INT32), col0);

    auto cy_col1 =
        cylon::VectorColumn<double>::Make("col1", cylon::DataType::Make(cylon::Type::DOUBLE), col1);

    std::shared_ptr<cylon::Table> output;
    status = cylon::Table::FromColumns(ctx, {cy_col0, cy_col1}, &output);

    REQUIRE((status.is_ok() && output->Columns() == 2 && output->Rows() == size));
  }
}