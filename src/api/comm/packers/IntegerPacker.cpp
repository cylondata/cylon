#include <array>
#include "DataPacker.h"

namespace twister::comm::packers {
    class IntegerPacker : public DataPacker<int> {

        void pack_to_buffer(int data, Buffer buffer, PackerStatus status) {
            bool completed = buffer.put_int(data);
            if (completed) {
                status.set_completed(true);
            }
        }
    };
}