#include <cstdint>
#include "DataPacker.h"

namespace twisterx::comm::packers {
    class IntegerArrayPacker : public DataPacker<int32_t *> {
        void pack_to_buffer(int32_t *data, Buffer buffer, PackerStatus status) {
            int32_t bytes_top_copy = std::min(status.left_to_process(), buffer.remaining());
            int32_t elements_to_copy = bytes_top_copy / sizeof(int32_t);
            int32_t already_copied = status.get_progress() / sizeof(int32_t);

            for (int32_t i = 0; i < elements_to_copy; i++) {
                buffer.put_int32(data[already_copied + i]);
            }

            status.add_to_progress(bytes_top_copy);
        }
    };
}