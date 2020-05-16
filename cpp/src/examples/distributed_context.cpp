#include <glog/logging.h>
#include <net/mpi/mpi_communicator.h>
#include <python/ctx/twisterx_context_wrap.h>
#include <table.hpp>

int main(int argc, char *argv[]) {
  std::string mpi_config = "mpi";
  auto ctx_wrap = twisterx::py::twisterx_context_wrap(mpi_config);
  auto ctx = ctx_wrap.getInstance();
  std::cout << "Hello World , Rank " << ctx->GetRank() << ", Size " << ctx->GetWorldSize() << std::endl ;
  ctx->Finalize();
  return 0;
}

