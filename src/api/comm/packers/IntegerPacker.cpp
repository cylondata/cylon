#include <array>
#include "DataPacker.h"

namespace twisterx::comm::packers {
    class IntegerPacker : public DataPacker<int32_t> {

        void pack_to_buffer(int32_t data, Buffer buffer, PackerStatus status) {
            bool completed = buffer.put_int32(data);
            if (completed) {
                status.add_to_progress(sizeof(int32_t));
            }
        }
    };
}