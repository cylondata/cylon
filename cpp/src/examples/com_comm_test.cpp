#include <ctx/cylon_context.hpp>
#include <net/mpi/mpi_communicator.hpp>
#include <table.hpp>
#include <ops/union_op.hpp>

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

  auto union_config = std::make_shared<cylon::UnionOpConfig>();

  auto union_op = cylon::UnionOp(std::shared_ptr<cylon::CylonContext>(ctx),
                                 table1_arr->schema(), 0, [](int x) {
        return 0;
      }, cb, union_config);
  LOG(INFO) << "Created  op";
  union_op.InsertTable(0, table1);
  union_op.InsertTable(1, table2);
  union_op.FinalizeInputs();

  while (!union_op.IsComplete()) {
    union_op.Progress();
  }

  union_op.Finalize();

  ctx->Finalize();
  return 0;
}
