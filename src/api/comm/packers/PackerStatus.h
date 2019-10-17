//
// Created by chathura on 10/17/19.
//

#ifndef TWISTERX_PACKERSTATUS_H
#define TWISTERX_PACKERSTATUS_H

namespace twisterx::comm::packers {
    class PackerStatus {
    private:
        int total;
        int progress;
        bool completed = false;

    public:
        int get_total() const {
            return total;
        }

        void set_completed(bool completed) {
            PackerStatus::completed = completed;
        }

        bool is_completed() {
            return PackerStatus::completed;
        }

        void set_total(int total) {
            PackerStatus::total = total;
        }

        int get_progress() const {
            return progress;
        }

        void set_progress(int progress) {
            PackerStatus::progress = progress;
        }
    };
}

#endif //TWISTERX_PACKERSTATUS_H
