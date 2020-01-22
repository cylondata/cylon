//
// Created by chathura on 10/22/19.
//

#ifndef TWISTERX_GATHERTEST_H
#define TWISTERX_GATHERTEST_H

#include <iostream>
#include <functional>
#include <queue>
#include "../Communicator.h"
#include "../Receiver.h"
#include "../packers/PackerStatus.h"

namespace twisterx::comm::op {

    template<class T>
    class GatherTest : public Receiver {
    private:
        int32_t op_id;
        Communicator *communicator;
        int32_t target;
        std::function<void(std::vector<std::pair<int32_t, T *>>)> gather_receiver;

        std::unordered_map<int32_t, twisterx::comm::packers::PackerStatus<T> *> gather_holders;

        twisterx::comm::packers::DataPacker<T> data_packer;

    public:
        GatherTest(Communicator &communicator, int32_t target,
                   std::function<void(std::vector<std::pair<int32_t, T *>>)> gather_receiver)
                : gather_receiver(gather_receiver) {
            this->op_id = communicator.next_op_id();
            this->communicator = &communicator;
            communicator.register_receiver(this->op_id, this);
            this->target = target;
            this->gather_receiver = gather_receiver;
        }

        void progress() {
            this->communicator->progress();
        }

        bool gather(T *data, size_t size) {
            if (this->communicator->get_worker_id() == this->target) {
                if (this->gather_holders[this->target] == nullptr) {
                    this->gather_holders[this->target] = new twisterx::comm::packers::PackerStatus<T>(
                            size, data, size
                    );
                    return true;
                } else {
                    //previous gather in progress
                    return false;
                }
            }
            return this->communicator->send_message(data, size, this->target, this->op_id);
        }

        void compile_gather_message() {
            bool gather_complete = true;
            for (int32_t worker_id = 0; worker_id < this->communicator->get_world_size(); worker_id++) {
                if (this->gather_holders[worker_id] == nullptr || !this->gather_holders[worker_id]->is_completed()) {
                    gather_complete = false;
                    break;
                }
            }

            if (gather_complete) {
                std::vector<std::pair<int32_t, T *>> gathered_data;
                for (int32_t worker_id = 0; worker_id < this->communicator->get_world_size(); worker_id++) {
                    gathered_data.push_back(
                            std::pair<int32_t, T *>(worker_id, this->gather_holders[worker_id]->get_data()));
                }
                std::cout << "gather completed" << std::endl;
                this->gather_receiver(gathered_data);
                this->gather_holders.clear();
            }
        }

        bool receive(int32_t source, Buffer *buffer) {
            std::cout << "Received from : " << source << std::endl;

            auto packer_status = this->gather_holders[source];
            if (packer_status == nullptr) {
                size_t data_size = buffer->get_int32();
                this->gather_holders[source] = new twisterx::comm::packers::PackerStatus<T>(
                        data_size, data_packer.get_new_data_holder(data_size)
                );
            }

            if (!this->gather_holders[source]->is_completed()) {
                this->data_packer.unpack_from_buffer(this->gather_holders[source], buffer);
                this->compile_gather_message();
                return true;
            }
            return false;
        }
    };


}

#endif //TWISTERX_GATHERTEST_H
