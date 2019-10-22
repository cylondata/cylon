//
// Created by chathura on 10/22/19.
//

#ifndef TWISTERX_MESSAGE_H
#define TWISTERX_MESSAGE_H

#include "../Buffer.h"

namespace twisterx::comm::messages {
    class Message {
    public:
        virtual bool offer_buffer(Buffer *buffer) = 0;

        virtual int32_t get_tag() = 0;
    };
}

#endif //TWISTERX_MESSAGE_H
