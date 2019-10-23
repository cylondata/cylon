//
// Created by chathura on 10/22/19.
//

#ifndef TWISTERX_RECEIVER_H
#define TWISTERX_RECEIVER_H

#include "Buffer.h"

namespace twisterx::comm {
    class Receiver {
    public:
        virtual bool receive(int32_t source, Buffer *buffer) = 0;
    };
}

#endif //TWISTERX_RECEIVER_H
