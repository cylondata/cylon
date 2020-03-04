#include "arrow_join.hpp"
#include "../join/join.hpp"

namespace twisterx {
	ArrowJoin::ArrowJoin(int worker_id, const std::vector<int> &source, const std::vector<int> &targets, int leftEdgeId,
											 int rightEdgeId, twisterx::JoinCallback *callback, std::shared_ptr<arrow::Schema> schema,
											 arrow::MemoryPool *pool) {
	  joinCallBack_ = callback;
		leftCallBack_ = std::make_shared<AllToAllCallback>(&leftTables_);
		rightCallBack_ = std::make_shared<AllToAllCallback>(&rightTables_);
		leftAllToAll_ =
				std::make_shared<ArrowAllToAll>(worker_id, source, targets, leftEdgeId, leftCallBack_, schema, pool);
		rightAllToAll_ =
				std::make_shared<ArrowAllToAll>(worker_id, source, targets, rightEdgeId, rightCallBack_, schema, pool);
	}

	bool ArrowJoin::isComplete() {
		bool left = leftAllToAll_->isComplete();
		bool right = rightAllToAll_->isComplete();

		if (left && right) {
			LOG(INFO) << "Received everything to join";
			std::shared_ptr<arrow::Table> joined_table;
			arrow::Status status = join::join(leftTables_, rightTables_, (int64_t)0, (int64_t)0,
								 join::JoinType::INNER, join::JoinAlgorithm::SORT,
								 &joined_table,
								 arrow::default_memory_pool());
			if (status != arrow::Status::OK()) {
        LOG(FATAL) << "Failed to join - error: " << status.CodeAsString();
        return true;
			}
			joinCallBack_->onJoin(joined_table);
			// join
			return true;
		}
		return false;
	}

	AllToAllCallback::AllToAllCallback(std::vector<std::shared_ptr<arrow::Table>> *table) {
		tables_ = table;
	}

	bool AllToAllCallback::onReceive(int source, std::shared_ptr<arrow::Table> table) {
		tables_->push_back(table);
    return true;
	}
}



