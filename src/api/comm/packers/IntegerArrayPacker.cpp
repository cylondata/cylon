#include <cstdint>
#include "DataPacker.h"

namespace twisterx::comm::packers {
    class IntegerArrayPacker : public DataPacker<int32_t *> {
        void pack_to_buffer(int32_t *data, Buffer buffer, PackerStatus status) {
            int32_t to_copy = status.left_to_process();
            for (int32_t i = 0; i < to_copy; i++) {
                buffer.put_int32(data[status.get_progress() + i]);
            }
            bool completed = buffer.put_int32(data);
            if (completed) {
                status.set_completed(true);
            }
        }
    };
}