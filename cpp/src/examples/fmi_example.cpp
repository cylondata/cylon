/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <glog/logging.h>

#include <cylon/net/fmi/fmi_communicator.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/table.hpp>
#include "thridparty/fmi/utils/DirectBackend.hpp"
#include "examples/example_utils.hpp"

#include <arrow/api.h>
#include <arrow/ipc/api.h>


#define CHECK_STATUS(status, msg) \
  if (!status.is_ok()) {          \
    LOG(ERROR) << msg << " " << status.get_msg(); \
    ctx->Finalize();              \
    return 1;                     \
  }

#define CHECK_ARROW_EQUAL(expected, received)                                 \
  do {                                                                         \
    const auto& exp_ = (expected);                                              \
    const auto& rec_ = (received);                                              \
    INFO("Expected: " << exp_->ToString() << "\nReceived: " << rec_->ToString());\
    REQUIRE(exp_->Equals(*rec_));                                                \
  } while(0)
static constexpr int kCount = 5000000;
static constexpr double kDup = 0.9;


std::shared_ptr<arrow::Array> ArrayFromJSON(const std::shared_ptr<arrow::DataType> &type,
                                            std::string_view json) {
    const auto &res = arrow::ipc::internal::json::ArrayFromJSON(type, json);
    return res.ValueOrDie();
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        LOG(ERROR) << "There should be an argument for directory";
        return 1;
    }

    if (argc < 3) {
        LOG(ERROR) << "There should be an argument for rank";
        return 1;
    }

    if (argc < 4) {
        LOG(ERROR) << "There should be an argument for worldsize";
        return 1;
    }

    if (argc < 5) {
        LOG(ERROR) << "There should be an argument for commname";
        return 1;
    }

    if (argc < 6) {
        LOG(ERROR) << "There should be an argument for host";
        return 1;
    }

    if (argc < 7) {
        LOG(ERROR) << "There should be an argument for port";
        return 1;
    }

    if (argc < 8) {
        LOG(ERROR) << "There should be an argument for maxTimeout";
        return 1;
    }

    if (argc < 9) {
        LOG(ERROR) << "There should be an argument for nonblocking";
        return 1;
    }

    if (argc < 10) {
        LOG(ERROR) << "There should be an argument for redisHost";
        return 1;
    }

    if (argc < 11) {
        LOG(ERROR) << "There should be an argument for redisPort";
        return 1;
    }

    if (argc < 12) {
        LOG(ERROR) << "THere should be an argument for redisNamespace";
        return 1;
    }

    auto directory = std::string(argv[1]);

    auto rank = std::stoi(argv[2]);

    auto worldsize = std::stoi(argv[3]);

    auto com_name = std::string(argv[4]);

    auto host = std::string(argv[5]);

    auto port = std::stoi(argv[6]);

    auto maxTimout = std::stoi(argv[7]);

    auto nonblocking = std::stoi(argv[8]);

    auto redisHost = std::string(argv[9]);

    auto redisPort = std::stoi(argv[10]);

    auto redisNamespace = std::string(argv[11]);


    /*auto backend = std::make_shared<FMI::Utils::DirectBackend>();



    backend->withHost(host.c_str());//rendezvous host
    backend->withPort(port);//rendezvous port
    backend->withMaxTimeout(maxTimout); //max timeout for direct connect
    backend->setResolveBackendDNS(false);//resolve rendezvous ip address
    backend->setBlockingMode(nonblocking ? FMI::Utils::NONBLOCKING : FMI::Utils::BLOCKING);


    std::shared_ptr<FMI::Utils::Backends> base_backend = std::dynamic_pointer_cast<FMI::Utils::Backends>(backend);*/

    auto config = std::make_shared<cylon::net::FMIConfig>(rank, worldsize,
                                                          host.c_str(), port,
                                                          maxTimout, false, com_name,
                                                          nonblocking ? FMI::Utils::NONBLOCKING : FMI::Utils::BLOCKING,
                                                          true, redisHost.c_str(), redisPort, redisNamespace);

    /*auto config = std::make_shared<cylon::net::FMIConfig>(rank, worldsize,
                                                          host.c_str(), port,
                                                          maxTimout, false, com_name,
                                                          nonblocking ? FMI::Utils::NONBLOCKING : FMI::Utils::BLOCKING,
                                                          true);*/
    std::shared_ptr<cylon::CylonContext> ctx;

    if (!cylon::CylonContext::InitDistributed(config, &ctx).is_ok()) {
        return 1;
    }


    LOG(INFO) << "rank:" << ctx->GetRank() << " size:" << ctx->GetWorldSize();

    ctx->Barrier();

    const int modified_rank = ctx->GetRank() + 1;

    cylon::Status status;
    /*const std::string csv1 =  directory + "user_device_tm_" + std::to_string(modified_rank) + ".csv";
    const std::string csv2 = directory + "user_usage_tm_" + std::to_string(modified_rank) + ".csv";

    std::shared_ptr<cylon::Table> first_table, second_table, joined_table;


    status = cylon::FromCSV(ctx, csv1, first_table);
    CHECK_STATUS(status, "Reading csv1 failed!")

    status = cylon::FromCSV(ctx, csv2, second_table);
    CHECK_STATUS(status, "Reading csv2 failed!")*/
    std::shared_ptr<cylon::Table> first_table, second_table, joined_table;
    cylon::examples::create_two_in_memory_tables(kCount, kDup, ctx, first_table, second_table);

    //auto join_config = cylon::join::config::JoinConfig::InnerJoin(0, 3);
    cylon::join::config::JoinConfig join_config{cylon::join::config::JoinType::INNER, 0, 0,
                                       cylon::join::config::JoinAlgorithm::SORT, "l_", "r_"};

    status = cylon::DistributedJoin(first_table, second_table, join_config, joined_table);
    LOG(INFO) << "Status returned: " << status.get_code() << " msg: " <<status.get_msg();
    CHECK_STATUS(status, "Join failed!")

    LOG(INFO) << "First table had : " << first_table->Rows() << " and Second table had : "
              << second_table->Rows() << ", Joined has : " << joined_table->Rows();

    LOG(INFO) << "AllReduce Collective Test";


    using TestType = arrow::Int32Type;
    std::shared_ptr<arrow::DataType> type = arrow::TypeTraits<TestType>::type_singleton();

    auto rank2 = *arrow::MakeScalar(ctx->GetRank())->CastTo(type);

    auto base_arr =  ArrayFromJSON(type, "[1, 2, 3, 4]");
    // all reduce local sample histograms
    auto arr = arrow::compute::Multiply(base_arr, rank2)->make_array();
    auto col = cylon::Column::Make(std::move(arr));

    const auto &comm = ctx->GetCommunicator();

    auto multiplier = *arrow::MakeScalar((worldsize - 1) * worldsize / 2)->CastTo(type);
    auto exp = arrow::compute::Multiply(base_arr, multiplier)->make_array();

    std::shared_ptr<cylon::Column> res;
    CHECK_STATUS(comm->AllReduce(col, cylon::net::SUM, &res), "allreducefailed");

    const auto &rcv = res->data();

    LOG(INFO) << "AllReduce Result: " << rcv->ToString();



    ctx->Finalize();
    return 0;


}
