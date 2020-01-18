#include "channel.hpp"

namespace twisterx {
  /**
   * This class implements a MPI channel, when there is a message to be sent,
   * this channel sends a small message with the size of the previous message
   */
  class MPIChannel : public Channel {
  public:
    bool send(TxRequest *request);
  private:
  };
}