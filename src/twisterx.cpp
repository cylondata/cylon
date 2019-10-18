#include <iostream>
#include "api/worker/Worker.h"
#include "api/comm/Buffer.h"


class MyWorker : public twisterx::worker::Worker {

    void execute(Config config) override {
        std::cout << "Starting TwisterX program ..." << std::endl;

        int32_t size = 10;
        twisterx::comm::Buffer buffer(size);

        buffer.put_int32(size);
        buffer.put_int32(size + 1);
        buffer.flip();

        int32_t read = buffer.get_int32();
        int32_t read2 = buffer.get_int32();

        int32_t go;

        std::cout << "read : " << read << read2 << std::endl;

        buffer.clear();
    }

};

int main() {
    MyWorker myWorker;
    myWorker.init(NULL, NULL);
    myWorker.start();
}
