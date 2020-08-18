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
 public:
  virtual void OnResult(int tag, std::shared_ptr<cylon::Table> table) = 0;
};

class Op {
 private:
  int id;
  std::unordered_map<int, std::queue<std::shared_ptr<cylon::Table>> *> queues{};

  // this count should be increased when adding table to the queues,
  // and should be decrease when removing a table from the queue
  int inputs_count = 0;
  std::unordered_map<int, cylon::Op *> children{};
  std::shared_ptr<ResultsCallback> callback;
  std::function<int(int)> router;

  std::queue<std::shared_ptr<cylon::Table>> *GetQueue(int tag);
  Op *GetChild(int tag);

 protected:
  std::shared_ptr<cylon::CylonContext> ctx_;

 public:
  Op(std::shared_ptr<cylon::CylonContext> ctx,
     int id, std::function<int(int)> router,
     std::shared_ptr<ResultsCallback> callback);

  void DrainQueueToChild(int queue, int child, int tag);

  /**
   * If this Op is a leaf op, send the table to the callback
   * @param tag tag of the table
   * @param table pointer to the table
   * @return true if this op is a leaf, false otherwise
   */
  bool TerminalCheck(int tag, std::shared_ptr<cylon::Table> table);

  /**
  * Inserts the given table to the input Queue of every child
  * @param tag tag of the table
  * @param table pointer to the table
  */
  void InsertToAllChildren(int tag, std::shared_ptr<cylon::Table> table);

  /**
   * Inserts the table to the input queue of a specific child
   * @param tag tag of the table
   * @param child target child ID
   * @param table pointer to the table
   */
  void InsertToChild(int tag, int child, std::shared_ptr<cylon::Table> table);

  /**
   * This function will be used by parent Ops to insert a table to the input queue
   * @param tag
   * @param table
   */
  void InsertTable(int tag, std::shared_ptr<cylon::Table> table);

  /**
   * This function defines the execution logic of this op. It can be either a computation or a communication
   * @param tag
   * @param table
   */
  virtual void execute(int tag, std::shared_ptr<Table> table) = 0;

  /**
   * This function depends on inputs_count to determine whether Op has any input data to proceed
   * @return
   */
  bool Ready();

  void Progress();

  bool IsComplete();

  ~Op();

  cylon::Op *AddChild(cylon::Op *child);
};
}
#endif //CYLON_SRC_CYLON_OPS_PARALLEL_OP_H_
