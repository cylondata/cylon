#ifndef TWISTERX_TYPE_H
#define TWISTERX_TYPE_H

#include <vector>

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
    virtual void * merge(std::vector<void *> buffers) = 0;
  };
}

#endif