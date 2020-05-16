#include <glog/logging.h>
#include <net/mpi/mpi_communicator.h>
#include <ctx/twisterx_context.h>
#include <table.hpp>

int main(int argc, char *argv[]) {

  auto mpi_config = new twisterx::net::MPIConfig();
  auto ctx = twisterx::TwisterXContext::InitDistributed(mpi_config);

  std::shared_ptr<twisterx::Table> table1, table2, joined;
  std::string join_file = "/tmp/test_join_records_1000000_columns_2.csv";
  auto status1 = twisterx::Table::FromCSV(join_file, &table1);
  auto status2 = twisterx::Table::FromCSV(join_file, &table2);

  table1->DistributedJoin(ctx, table2,
               twisterx::join::config::JoinConfig::InnerJoin(0, 0),
               &joined);
  ctx->Finalize();
  return 0;
}
