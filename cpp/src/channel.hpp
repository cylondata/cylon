#ifndef TWISTERX_CHANNEL_H
#define TWISTERX_CHANNEL_H

#include <vector>
#include <memory>
#include <cstring>

namespace twisterx {
  /**
 * When a buffer is inserted, we need to return a reference to that buffer
 */
  struct TxRequest {
    void * buffer{};
    int length{};
    int target;
    int header[6] = {};
    int headerLength{};

    TxRequest(int tgt, void *buf, int len) {
      target = tgt;
      buffer = buf;
      length = len;
    }

    TxRequest(int tgt, void *buf, int len, int * head, int hLength) {
      target = tgt;
      buffer = buf;
      length = len;
      // we are copying the header
      memcpy(&header[0], head, hLength * sizeof(int));
      headerLength = hLength;
    }

    explicit TxRequest(int tgt) {
      target = tgt;
    }

    ~TxRequest() {
      // LOG(INFO) << "Delete the request with address" << buffer;
      buffer = nullptr;
    };
  };

  /**
   * When a send is complete, this callback is called by the channel, it is the responsibility
   * of the operations to register this callback
   */
   class ChannelSendCallback {
   public:
     virtual void sendComplete(std::shared_ptr<TxRequest> request) = 0;

     virtual void sendFinishComplete(std::shared_ptr<TxRequest> request) = 0;
   };

  /**
   * When a receive is complete, this method is called
   */
   class ChannelReceiveCallback {
   public:
     virtual void receivedData(int receiveId, void *buffer, int length) = 0;

     virtual void receivedHeader(int receiveId, int finished, int * header, int headerLength) = 0;
   };

  /**
   * This is an interface to send messages using a particular channel, a channel
   * can be based on MPI, it can be a TCP channel or a UCX channel etc
   */
  class Channel {
  public:
    /**
     * Initialize the channel with the worker ids from which we are going to receive
     *
     * @param receives these are the workers we are going to receive from
     */
    virtual void init(int edge, const std::vector<int>& receives, const std::vector<int>& sendIds,
                      ChannelReceiveCallback * rcv, ChannelSendCallback * send) = 0;
    /**
     * Send the request
     * @param request the request containing buffer, destination etc
     * @return if the request is accepted to be sent
     */
    virtual int send(std::shared_ptr<TxRequest> request) = 0;

    /**
     * Inform the finish to the target
     * @param request the request
     * @return -1 if not accepted, 1 if accepted
     */
    virtual int sendFin(std::shared_ptr<TxRequest> request) = 0;

    /**
     * This method needs to be called to progress the send
     */
    virtual void progressSends() = 0;

    /**
     * This method needs to be called to progress the receives
     */
    virtual void progressReceives() = 0;

    /**
     * Close the channel and clear any allocated memory by the channel
     */
    virtual void close() = 0;

    virtual ~Channel() = default;
  };
}

#endif