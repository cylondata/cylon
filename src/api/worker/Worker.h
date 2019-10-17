#ifndef TWISTERX_WORKER_H
#define TWISTERX_WORKER_H

#include <mpi.h>
#include "../config/Config.h"

using namespace twister::config;

namespace twister::worker {
    class Worker {
    protected:
        int worker_id{};
        int world_size{};

    public:
        void init(int argc, char *argv[]) {
            int provided;
            MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
            if (provided < MPI_THREAD_MULTIPLE) {
                printf("ERROR: The MPI library does not have full thread support\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            MPI_Comm_size(MPI_COMM_WORLD, &this->world_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &this->worker_id);
        }

        void start() {
            // resolve configs
            Config config;
            this->execute(config);
            MPI_Finalize();
        }

        virtual void execute(Config config) = 0;

        bool isMaster() {
            return this->worker_id == 0;
        }
    };
}

#endif //TWISTERX_WORKER_H
