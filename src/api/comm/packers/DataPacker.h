#ifndef TWISTERX_DATAPACKER_H
#define TWISTERX_DATAPACKER_H

#include "../Buffer.h"
#include "PackerStatus.h"

namespace twisterx::comm::packers {
    template<class T>
    class DataPacker {
        void pack_to_buffer(T data, Buffer buffer, PackerStatus status);
    };
}

#endif //TWISTERX_DATAPACKER_H
