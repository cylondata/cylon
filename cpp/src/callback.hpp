#ifndef TWISTERX_CALLBACK_H
#define TWISTERX_CALLBACK_H

namespace twisterx {

  class ReceiveCallback {
  public:
    /**
     * This function is called when a data is received
     * @param source the source
     * @param buffer the buffer allocated by the system, we need to free this
     * @param length the length of the buffer
     * @return true if we accept this buffer
     */
    virtual bool onReceive(int source, void * buffer, int length) = 0;

    /**
     * Receive the header, this happens before we receive the actual data
     * @param source the source
     * @param buffer the header buffer, which can be 6 integers
     * @param length the length of the integer array
     * @return true if we accept the header
     */
    virtual bool onReceiveHeader(int source, int * buffer, int length) = 0;
  };

}
#endif