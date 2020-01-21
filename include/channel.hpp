#include "request.hpp"

#include <vector>

namespace twisterx {
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
    virtual void init(int edge, const std::vector<int>& receives, const std::vector<int>& sendIds) = 0;
    /**
     * Send the request
     * @param request the request containing buffer, destination etc
     * @return if the request is accepted to be sent
     */
    virtual bool send(TxRequest *request) = 0;

    /**
     * This method needs to be called to progress the send
     */
    virtual void progressSends() = 0;

    /**
     * This method needs to be called to progress the receives
     */
    virtual void progressReceives() = 0;
  };
}