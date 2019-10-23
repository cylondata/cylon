//
// Created by chathura on 10/22/19.
//

#ifndef TWISTERX_GATHERTEST_H
#define TWISTERX_GATHERTEST_H

#import <iostream>
#include "../Communicator.h"
#include "../Receiver.h"

namespace twisterx::comm::op {

    template<class T>
    class GatherTest : public Receiver {
    private:
        int32_t op_id;
        Communicator *communicator;
        int32_t target;
    public:
        GatherTest(Communicator &communicator, int32_t target) {
            this->op_id = communicator.next_op_id();
            this->communicator = &communicator;
            communicator.register_receiver(this->op_id, this);
            this->target = target;
        }

        void progress() {
            this->communicator->progress();
        }

        bool gather(T *data, size_t size) {
            if (this->communicator->get_worker_id() == this->target) {
                return true;
            }
            return this->communicator->send_message(data, size, this->target, this->op_id);
        }

        void receive(int32_t source, Buffer *buffer) {
            std::cout << "Received from : " << source << std::endl;
        }
    };
}

#endif //TWISTERX_GATHERTEST_H
