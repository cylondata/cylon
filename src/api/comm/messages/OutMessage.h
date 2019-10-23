//
// Created by chathura on 10/22/19.
//

#ifndef TWISTERX_OUTMESSAGE_H
#define TWISTERX_OUTMESSAGE_H

#include "../packers/DataPacker.h"
#include "Message.h"
#include "../packers/PackerStatus.h";

namespace twisterx::comm::messages {
    template<class T>
    class OutMessage : public Message {

    private:
        twisterx::comm::packers::DataPacker<T> *dataPacker;
        twisterx::comm::packers::PackerStatus<T> *packerStatus;
        int32_t destination;
        int32_t op_id;
        bool first_buffer = true;
    public:
        OutMessage(T *data, size_t size,
                   int32_t destination, int32_t op_id) {
            this->destination = destination;
            this->op_id = op_id;
            this->packerStatus = new packers::PackerStatus<T>(size, data);
            this->dataPacker = new packers::DataPacker<T>();
        }

        bool offer_buffer(Buffer *buffer) {
            buffer->put_int32(this->op_id);
            if (this->first_buffer) {
                buffer->put_int32(this->packerStatus->get_total()); //total size
                this->first_buffer = false;
            }
            this->dataPacker->pack_to_buffer(this->packerStatus, buffer);
            return this->packerStatus->is_completed();
        }

        int32_t get_tag() {
            return this->destination;
        }
    };
}

#endif //TWISTERX_OUTMESSAGE_H
