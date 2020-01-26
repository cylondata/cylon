#ifndef TWISTERX_MPI_CHANNEL_H
#define TWISTERX_MPI_CHANNEL_H

#include "channel.hpp"

#include <vector>
#include <unordered_map>
#include <queue>
#include <mpi.h>

namespace twisterx {
  enum SendStatus {
    SEND_INIT = 0,
    SEND_LENGTH_POSTED = 1,
    SEND_POSTED = 2,
    SEND_FINISH = 3
  };

  enum ReceiveStatus {
    RECEIVE_INIT = 0,
    RECEIVE_LENGTH_POSTED = 1,
    RECEIVE_POSTED = 2,
    RECEIVED_FIN = 3
  };

  /**
   * Keep track about the length buffer to receive the length first
   */
  struct PendingSend {
    //  for a given send we will use the length buffer or the request
    int headerBuf[2];
    std::queue<TxRequest *> pendingData;
    SendStatus status = SEND_INIT;
    MPI_Request request{};
    // the current send, if it is a actual send
    TxRequest * currentSend{};
  };

  struct PendingReceive {
    int headerBuf[2];
    int receiveId{};
    void * data{};
    int length{};
    ReceiveStatus status = RECEIVE_INIT;
    MPI_Request  request{};
  };

  /**
   * This class implements a MPI channel, when there is a message to be sent,
   * this channel sends a small message with the size of the next message. This allows the other side
   * to post the network buffer to receive the message
   */
  class MPIChannel : public Channel {
  public:
    /**
     * Initialize the channel
     *
     * @param receives receive from these ranks
     */
    void init(int edge, const std::vector<int>& receives, const std::vector<int>& sendIds,
              ChannelReceiveCallback * rcv, ChannelSendCallback * send) override;

    /**
    * Send the message to the target.
    *
    * @param request the request
    * @return true if accepted
    */
    int send(TxRequest *request) override;

    /**
    * Send the message to the target.
    *
    * @param request the request
    * @return true if accepted
    */
    int sendFin(TxRequest *request) override;

    /**
     * This method, will send the messages, It will first send a message with length and then
     */
    void progressSends() override;

    /**
     * Progress the pending receivers
     */
    void progressReceives() override;

    void close() override;

  private:
    int edge;
    // keep track of the length buffers for each receiver
    std::unordered_map<int, PendingSend *> sends;
    // keep track of the posted receives
    std::unordered_map<int, PendingReceive *> pendingReceives;
    // we got finish requests
    std::unordered_map<int, TxRequest *> finishRequests;
    // receive callback function
    ChannelReceiveCallback * rcv_fn;
    // send complete callback function
    ChannelSendCallback * send_comp_fn;

    /**
     * Send finish request
     * @param x the target, pendingSend pair
     */
    void sendFinishRequest(const std::pair<const int, PendingSend *> &x) const;

    /**
     * Send the length
     * @param x the target, pendingSend pair
     */
    void sendLength(const std::pair<const int, PendingSend *> &x) const;
  };
}

#endif