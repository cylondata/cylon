#include <glog/logging.h>
#include <net/mpi/mpi_communicator.h>
#include <python/ctx/twisterx_context_wrap.h>

int main(int argc, char *argv[]) {
  std::string mpi_config = "mpi";
  auto ctx_wrap = twisterx::py::twisterx_context_wrap(mpi_config);
  auto ctx = ctx_wrap.getInstance();
  std::cout << "Hello World , Rank " << ctx_wrap.GetRank() << ", Size " << ctx_wrap.GetWorldSize() << std::endl;
  ctx_wrap.Finalize();
  return 0;
}

