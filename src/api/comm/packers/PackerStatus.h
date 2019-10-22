#ifndef TWISTERX_PACKERSTATUS_H
#define TWISTERX_PACKERSTATUS_H

namespace twisterx::comm::packers {
    template<class T>
    class PackerStatus {
    private:
        int32_t total;
        int32_t progress;
        T *data;

    public:
        PackerStatus(int32_t total, T *data) {
            this->total = total;
            this->data = data;
        }

        [[nodiscard]] int32_t get_total() const {
            return total;
        }

        bool is_completed() {
            return total == progress;
        }

        [[nodiscard]] int32_t get_progress() const {
            return progress;
        }

        void add_to_progress(int32_t delta) {
            progress += delta;
        }

        int32_t left_to_process() {
            return total - progress;
        }

        T *get_data() const {
            return data;
        }

        void set_data(T *new_data) {
            this->data = new_data;
        }
    };
}

#endif //TWISTERX_PACKERSTATUS_H
