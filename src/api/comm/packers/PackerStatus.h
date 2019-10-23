#ifndef TWISTERX_PACKERSTATUS_H
#define TWISTERX_PACKERSTATUS_H

namespace twisterx::comm::packers {
    template<class T>
    class PackerStatus {
    private:
        size_t total;
        size_t progress = 0;
        T *data;

    public:
        PackerStatus(size_t total, T *data) {
            this->total = total;
            this->data = data;
        }

        PackerStatus(size_t total, T *data, size_t progress) {
            this->total = total;
            this->data = data;
            this->progress = progress;
        }

        [[nodiscard]] size_t get_total() const {
            return total;
        }

        bool is_completed() {
            return total == progress;
        }

        [[nodiscard]] size_t get_progress() const {
            return progress;
        }

        void add_to_progress(size_t delta) {
            progress += delta;
        }

        size_t left_to_process() {
            return total - progress;
        }

        T *get_data() const {
            return data;
        }

        void set_data(T *new_data) {
            this->data = new_data;
        }

        PackerStatus(PackerStatus &) = delete;
    };
}

#endif //TWISTERX_PACKERSTATUS_H
