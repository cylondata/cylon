#include <cylon/net/comm_operations.hpp>
#include <cylon/net/ops/base_ops.hpp>
#include <ucc/api/ucc.h>
#include <ucc/api/ucc_def.h>

namespace cylon {
namespace ucc {

class UccTableAllgatherImpl : public net::TableAllgatherImpl {
 public:
  UccTableAllgatherImpl(ucc_team_h ucc_team, ucc_context_h ucc_context, int world_size);
  ~UccTableAllgatherImpl() override;

  void Init(int num_buffers) override;

  Status AllgatherBufferSizes(const int32_t *send_data, int num_buffers,
                              int32_t *rcv_data) const override;

  Status IallgatherBufferData(
      int buf_idx, const uint8_t *send_data, int32_t send_count,
      uint8_t *recv_data, const std::vector<int32_t> &recv_count,
      const std::vector<int32_t> &displacements) override;

  Status WaitAll(int num_buffers) override;

 private:
  ucc_team_h ucc_team_;
  ucc_context_h ucc_context_;
  std::vector<ucc_coll_req_h> requests_;
  std::vector<ucc_coll_args_t> args_;
  std::vector<std::vector<uint64_t>> counts_;
  std::vector<std::vector<uint64_t>> displacements_;
  int world_size;
};

}  // namespace ucc
}  // namespace cylon