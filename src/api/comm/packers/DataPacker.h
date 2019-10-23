#ifndef TWISTERX_DATAPACKER_H
#define TWISTERX_DATAPACKER_H

#include "../Buffer.h"
#include "PackerStatus.h"

namespace twisterx::comm::packers {
    template<class T>
    class DataPacker {
        //todo can be singleton for each data type
    public:
        void pack_to_buffer(PackerStatus<T> *status, Buffer *buffer) {
            int32_t bytes_to_copy = std::min(status->left_to_process(), buffer->remaining());
            //round down to nearest sizeof(T)
            bytes_to_copy = (bytes_to_copy / sizeof(T)) * sizeof(T);

            int32_t already_copied = status->get_progress() / sizeof(T);

            buffer->put(status->get_data() + already_copied, bytes_to_copy);
            status->add_to_progress(bytes_to_copy);
        }

        void unpack_from_buffer(PackerStatus<T> *status, Buffer *buffer) {
            T *data = status->get_data();

            int32_t bytes_to_copy = std::min(status->left_to_process(), buffer->remaining());
            //round down to nearest sizeof(T)
            bytes_to_copy = (bytes_to_copy / sizeof(T)) * sizeof(T);

            int32_t already_copied = status->get_progress() / sizeof(T);

            buffer->get(data + already_copied, bytes_to_copy);
            status->add_to_progress(bytes_to_copy);
        }

        T *get_new_data_holder(size_t size) {
            auto *data_holder = new T[size];
            return data_holder;
        }
    };
}

#endif //TWISTERX_DATAPACKER_H
