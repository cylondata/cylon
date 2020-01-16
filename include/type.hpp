
namespace twisterx {
  /**
   * Defines a schema
   */
  class Schema {
    /**
     * Merge the first buffer, second buffer and produce a new buffer. This new buffer needs to be
     * @param first
     * @param second
     * @return
     */
    void * merge(std::vector<void *> buffers);
  }
}