#ifndef CYLON_SRC_CYLON_OPS_PARALLEL_OP_H_
#define CYLON_SRC_CYLON_OPS_PARALLEL_OP_H_

#include <memory>
#include <table.hpp>

/**
 * Assumptions
 * 1. Queue, Map lookups will never fail
 */
namespace cylon {
class ResultsCallback {
  virtual void OnResult(int tag, std::shared_ptr<cylon::Table> table);
};

class OpConfig {
  std::unordered_map<std::string, std::string> config{};
  OpConfig *AddConfig(const std::string &key, const std::string &value);
  std::string GetConfig(const std::string &key, const std::string &def = "");
};

class Op {
 protected:
  int id;
  std::unordered_map<int, std::queue<std::shared_ptr<cylon::Table>> *> queues{};
  int inputs_count = 0;
  std::unordered_map<int, cylon::Op *> children{};
  std::shared_ptr<ResultsCallback> callback;
  std::function<int(int)> router;

  Op *GetChild(int tag);

  void DrainQueueToChild(int queue, int child, int tag);

  void InsertToAllChildren(int tag, std::shared_ptr<cylon::Table> table);

  std::queue<std::shared_ptr<cylon::Table>> *GetQueue(int tag);

 public:
  Op(int id, std::function<int(int)> router, std::shared_ptr<ResultsCallback> callback);

  void insert(int tag, std::shared_ptr<cylon::Table> table);

  /**
   * This is the logic of this op
   */
  virtual void execute(int tag, std::shared_ptr<Table> table) = 0;

  virtual bool ready() = 0;

  void init(std::shared_ptr<cylon::CylonContext> ctx, std::shared_ptr<OpConfig> op_config);

  void progress();

  bool isComplete();

  ~Op();

  cylon::Op *AddChild(cylon::Op *child);
};
}
#endif //CYLON_SRC_CYLON_OPS_PARALLEL_OP_H_
