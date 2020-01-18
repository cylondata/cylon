#include<vector>
#include<map>

namespace twisterx {
  /**
   * The all to all communication. We insert values and wait until it completes
   */
  class AllToAll {
  public:
    /**
     * Constructor
     * @param worker_id
     * @param all_workers
     * @return
     */
    AllToAll(int worker_id, std::vector<int> source, std::vector<int> targets, int edgeId);

    /**
     * Insert a buffer to be sent, if the buffer is accepted return true
     *
     * @param buffer the buffer to send
     * @param length the length of the message
     * @param target the target to send the message
     * @return true if the buffer is accepted
     */
    bool insert(void *buffer, int length, int target);

    /**
     * Check weather the operation is complete, this method needs to be called until the operation is complete
     * @return true if the operation is complete
     */
    bool is_complete();
  private:
    int worker_id;                 // the worker id
    std::vector<int> sources;  // the list of all the workers
    std::vector<int> targets;  // the list of all the workers
    int edge;
    std::map<int, std::vector<void *>> buffers;  // keep the buffers to send
    std::map<int, int>  message_sizes;           // buffer sizes to send

    int current_send_id;           // the current send id
    int current_receive_id;        // the current receive id
  }
}
