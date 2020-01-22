#ifndef TWISTERX_ALL_TO_ALL_H
#define TWISTERX_ALL_TO_ALL_H

#include<vector>
#include<map>
#include<queue>

#include "channel.hpp"
#include "callback.hpp"

namespace twisterx {
  /**
   * The all to all communication. We insert values and wait until it completes
   */
  class AllToAll : public ChannelReceiveCallback, ChannelSendCallback {
  public:
    /**
     * Constructor
     * @param worker_id
     * @param all_workers
     * @return
     */
    AllToAll(int worker_id, const std::vector<int>& source, const std::vector<int>& targets, int edgeId,
        ReceiveCallback * callback);

    /**
     * Insert a buffer to be sent, if the buffer is accepted return true
     *
     * @param buffer the buffer to send
     * @param length the length of the message
     * @param target the target to send the message
     * @return true if the buffer is accepted
     */
    int insert(void *buffer, int length, int target);

    /**
     * Check weather the operation is complete, this method needs to be called until the operation is complete
     * @return true if the operation is complete
     */
    bool isComplete();

    /**
     * When this function is called, the operation finishes at both receivers and targets
     * @return
     */
    void finish();

    /**
     * We implement the receive complete callback from channel
     * @param receiveId
     * @param buffer
     * @param length
     */
    void receiveComplete(int receiveId, void *buffer, int length) override;

    /**
     * We implement the send callback from channel
     * @param request the original request, we can free it now
     */
    void sendComplete(TxRequest *request) override;

  private:
    int worker_id;                 // the worker id
    std::vector<int> sources;  // the list of all the workers
    std::vector<int> targets;  // the list of all the workers
    int edge;                  // the edge id we are going to use
    std::map<int, std::queue<TxRequest *>> buffers;  // keep the buffers to send
    std::map<int, int>  message_sizes;           // buffer sizes to send
    bool finishFlag = false;
    Channel * channel;             // the underlying channel
    ReceiveCallback * callback;    // after we receive a buffer we will call this function
  };
}

#endif