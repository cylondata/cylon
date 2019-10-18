//
// Created by chathura on 10/17/19.
//

#ifndef TWISTERX_PACKERSTATUS_H
#define TWISTERX_PACKERSTATUS_H

namespace twisterx::comm::packers {
    class PackerStatus {
    private:
        int32_t total;
        int32_t progress;

    public:
        PackerStatus(int32_t total) {
            this->total = total;
        }

        int32_t get_total() const {
            return total;
        }

        bool is_completed() {
            return total == progress;
        }

        int32_t get_progress() const {
            return progress;
        }

        void add_to_progress(int32_t delta) {
            progress += progress;
        }

        int32_t left_to_process() {
            return total - progress;
        }

    };
}

#endif //TWISTERX_PACKERSTATUS_H
