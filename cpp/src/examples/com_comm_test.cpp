#include <ctx/cylon_context.hpp>
#include <net/mpi/mpi_communicator.hpp>
#include <table.hpp>
#include <ops/dis_union_op.hpp>

int main(int argc, char *argv[]) {

  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  shared_ptr<cylon::Table> table1, table2, out;
  cylon::Table::FromCSV(ctx, "/home/chathura/Code/twisterx/cpp/data/csv1.csv", table1);
  cylon::Table::FromCSV(ctx, "/home/chathura/Code/twisterx/cpp/data/csv2.csv", table2);

  shared_ptr<arrow::Table> table1_arr, table2_arr;
  table1->ToArrowTable(table1_arr);
  table2->ToArrowTable(table2_arr);

  LOG(INFO) << "read table";
  class Cb : public cylon::ResultsCallback {
   public:
    virtual void OnResult(int tag, std::shared_ptr<cylon::Table> table) {
      LOG(INFO) << "Result received " << table->Rows();
    }
  };

  auto cb = std::make_shared<Cb>();

  auto union_config = std::make_shared<cylon::DisUnionOpConfig>();

  auto union_op = cylon::DisUnionOp(std::shared_ptr<cylon::CylonContext>(ctx),
                                    table1_arr->schema(), 0, cb, union_config);
  LOG(INFO) << "Created  op";
  union_op.InsertTable(0, table1);
  union_op.InsertTable(1, table2);

  while (!union_op.IsComplete()) {
    union_op.Progress();
  }

  ctx->Finalize();
  return 0;
}
