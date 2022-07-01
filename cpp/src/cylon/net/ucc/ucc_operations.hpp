#include <cylon/net/comm_operations.hpp>
#include <cylon/net/ops/base_ops.hpp>
#include <ucc/api/ucc.h>
#include <ucc/api/ucc_def.h>
#include <cylon/net/comm_operations.hpp>
namespace cylon {
namespace ucc {

class UccTableAllgatherImpl : public net::TableAllgatherImpl {
 public:
  UccTableAllgatherImpl(ucc_team_h ucc_team, ucc_context_h ucc_context, int rank, int world_size);
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
  std::vector<std::vector<uint32_t>> counts_;
  std::vector<std::vector<uint32_t>> displacements_;
  int rank;
  int world_size;
};

class UccAllReduceImpl : public net::AllReduceImpl {
public:
  ~UccAllReduceImpl() override = default;
  Status AllReduceBuffer(const void *send_buf, void *rcv_buf, int count,
                         const std::shared_ptr<DataType> &data_type,
                         net::ReduceOp reduce_op) const override;
  
  UccAllReduceImpl(ucc_team_h ucc_team, ucc_context_h ucc_context, int ws);

private:
  ucc_team_h ucc_team_;
  ucc_context_h ucc_context_;
  int world_size;
};

class UccTableGatherImpl : public net::TableGatherImpl {
public:
  UccTableGatherImpl(ucc_team_h ucc_team, ucc_context_h ucc_context, int rank, int world_size);
  void Init(int32_t num_buffers) override;

  Status GatherBufferSizes(const int32_t *send_data,
                                    int32_t num_buffers,
                                    int32_t *rcv_data,
                                    int32_t gather_root) const override;

  Status IgatherBufferData(int32_t buf_idx, const uint8_t *send_data,
                                   int32_t send_count, uint8_t *recv_data,
                                   const std::vector<int32_t> &recv_count,
                                   const std::vector<int32_t> &displacements,
                                   int32_t gather_root) override;

  Status WaitAll(int32_t num_buffers) override;

 private:
  std::vector<ucc_coll_req_h> requests_;
  std::vector<ucc_coll_args_t> args_;
  std::vector<std::vector<uint32_t>> counts_;
  std::vector<std::vector<uint32_t>> displacements_;
  ucc_team_h ucc_team_;
  ucc_context_h ucc_context_;
  int world_size;
  int rank;
};

}  // namespace ucc
}  // namespace cylon