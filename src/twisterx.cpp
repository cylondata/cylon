#include <iostream>
#include "api/worker/Worker.h"
#include "api/comm/Buffer.h"

using namespace twister::worker;
using namespace twister::comm;


class MyWorker : public Worker {

    void execute(Config config) override {
        std::cout << "Starting TwisterX program ..." << std::endl;

        int size = 10;
        Buffer buffer(size);

        buffer.put_int(size);
        buffer.flip();

        int read = buffer.get_int();

        std::cout << "read : " << read << std::endl;
    }

};

int main() {
    MyWorker myWorker;
    myWorker.init(0, NULL);
    myWorker.start();
}
