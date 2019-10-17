#include <iostream>
#include "api/worker/Worker.h"
#include "api/comm/Buffer.h"


class MyWorker : public twister::worker::Worker {

    void execute(Config config) override {
        std::cout << "Starting TwisterX program ..." << std::endl;

        int size = 10;
        twister::comm::Buffer buffer(size);

        buffer.put_int(size);
        buffer.put_int(size + 1);
        buffer.flip();

        int read = buffer.get_int();
        int read2 = buffer.get_int();

        std::cout << "read : " << read << read2 << std::endl;

        buffer.clear();
    }

};

int main() {
    MyWorker myWorker;
    myWorker.init(NULL, NULL);
    myWorker.start();
}
