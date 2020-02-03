//
// Created by skamburu on 1/29/20.
//

#ifndef TWISTERX_ARROW_H
#define TWISTERX_ARROW_H

#include <arrow/api.h>
#include "../all_to_all.hpp"

namespace twisterx {
  class ArrowAllToAll {
  public:
    /**
     * Constructor
     * @param worker_id
     * @param all_workers
     * @return
     */
    ArrowAllToAll(int worker_id, const std::vector<int> &source, const std::vector<int> &targets, int edgeId,
                  ReceiveCallback *callback);

    /**
     * Insert a buffer to be sent, if the buffer is accepted return true
     *
     * @param buffer the buffer to send
     * @param length the length of the message
     * @param target the target to send the message
     * @return true if the buffer is accepted
     */
    int insert(arrow::Table *arrow, int length, int target);

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
    void receiveComplete(int receiveId, void *buffer, int length);

    /**
     * We implement the send callback from channel
     * @param request the original request, we can free it now
     */
    void sendComplete(TxRequest *request);


    void receivedFinish(int receiveId);

    /**
     * Close the operation
     */
    void close();

  private:
    void sendFinishComplete(TxRequest *request);

    /**
     * The inderlying alltoall communication
     */
    AllToAll all;
  };
}
#endif //TWISTERX_ARROW_H
