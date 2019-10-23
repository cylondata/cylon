#include <iostream>
#include "api/worker/Worker.h"
#include "api/comm/Buffer.h"
#include "api/comm/packers/DataPacker.h"
#include "api/comm/packers/PackerStatus.h"
#include "api/comm/Communicator.h"
#include "api/comm/op/GatherTest.h"


class MyWorker : public twisterx::worker::Worker {

    void execute(Config config) {
        std::cout << "Starting TwisterX program ..." << std::endl;

        /*twisterx::comm::packers::DataPacker<int32_t> integerPacker;

        size_t size = 10;
        twisterx::comm::Buffer buffer(size);
        twisterx::comm::Buffer buffer2(size);

        int32_t data[] = {1, 2, 3};

        twisterx::comm::packers::PackerStatus<int32_t> send(sizeof(int32_t) * 3, data);
        integerPacker.pack_to_buffer(send, buffer);
        integerPacker.pack_to_buffer(send, buffer2);

        std::cout << send.is_completed() << std::endl;

        buffer.flip();
        buffer2.flip();

        auto *copy = new int[3];
        twisterx::comm::packers::PackerStatus<int32_t> recv(sizeof(int32_t) * 3, copy);
        integerPacker.unpack_from_buffer(recv, buffer);
        integerPacker.unpack_from_buffer(recv, buffer2);

        for (int i = 0; i < 3; i++) { std::cout << copy[i] << ","; }
        std::cout << std::endl;*/

        twisterx::comm::Communicator communicator(config,
                                                  this->worker_id, this->world_size);

        twisterx::comm::op::GatherTest<int32_t> gather(communicator, 0,
                                                       [](std::vector<std::pair<int32_t, int32_t *>> gathered_data) {
                                                           std::cout << "gather received, printing first few elements"
                                                                     << std::endl;
                                                           std::vector<std::pair<int32_t, int32_t *>>::iterator it;
                                                           for (it = gathered_data.begin();
                                                                it < gathered_data.end(); it++) {
                                                               std::cout << it->first << " : ";
                                                               for (int8_t i = 0; i < 3; i++) {
                                                                   std::cout << it->second[i] << ",";
                                                               }
                                                               std::cout << "..." << std::endl;
                                                           }
                                                       });

        auto *arr = new int32_t[10000];

        for (int i = 0; i < 10000; i++) {
            arr[i] = worker_id;
        }

        gather.gather(arr, sizeof(int32_t) * 10000);

        while (true)gather.progress();


        std::cout << "end of execute" << std::endl;
    }

};

int main() {
    MyWorker myWorker;
    myWorker.init(0, NULL);
    myWorker.start();
}
