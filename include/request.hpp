namespace twisterx {
  /**
   * When a buffer is inserted, we need to return a reference to that buffer
   */
  struct TxRequest {
    void * buffer;
    int length;
    int target;
    int edge;
    bool complete;
    /**
     * Channel specific holder
     */
    void * channel;
  };
}