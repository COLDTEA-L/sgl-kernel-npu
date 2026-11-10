// Public CCU API migration probe. No legacy CcuKernel/CcuRep C++ ABI is used.
#include <acl/acl.h>
#include <hccl/hccl.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_channel.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res.h>
#include <hcomm/ccu/ccu_launch.h>
#include <hcomm/ccu/ccu_primitives.hpp>
#include <hcomm/ccu/ccu_res.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ccu = AscendC::ccu;
namespace {
void Check(long status, const char* stage) {
    if (status != 0) throw std::runtime_error(std::string(stage) + " status=" + std::to_string(status));
}
#define CHECK(call) Check(static_cast<long>(call), #call)
#define EMIT(call) do { const auto status = (call); if (status != CcuResult::CCU_SUCCESS) return status; } while (0)

// One bidirectional direct channel. Channel variable slots and notification
// slots are separate resources; slots 0/1 publish the destination VA/token.
CcuResult ExchangeKernel(CcuKernelArg arg) {
    const auto channel = *static_cast<const ChannelHandle*>(arg);
    ccu::Variable input, output, inputToken, outputToken, bytes;
    EMIT(ccu::LoadArg(input, 0));
    EMIT(ccu::LoadArg(output, 1));
    EMIT(ccu::LoadArg(inputToken, 2));
    EMIT(ccu::LoadArg(outputToken, 3));
    EMIT(ccu::LoadArg(bytes, 4));
    auto remoteOutput = ccu::GetResByChannel<ccu::Variable>(channel, 0);
    auto remoteToken = ccu::GetResByChannel<ccu::Variable>(channel, 1);
    EMIT(ccu::WriteVariableWithNotify(channel, output, 0, 0));
    EMIT(ccu::WriteVariableWithNotify(channel, outputToken, 1, 1));
    EMIT(ccu::NotifyWait(channel, 0));
    EMIT(ccu::NotifyWait(channel, 1));
    ccu::LocalAddr src;
    src.addr = input;
    src.token = inputToken;
    ccu::RemoteAddr dst;
    dst.addr = remoteOutput;
    dst.token = remoteToken;
    ccu::Event done;
    CCU_IF(bytes != 0) {
        EMIT(ccu::Write(channel, dst, src, bytes, done));
        EMIT(ccu::EventWait(done));
    }
    // Zero-byte ranks still publish metadata and complete the handshake.
    EMIT(ccu::NotifyRecord(channel, 2));
    EMIT(ccu::NotifyWait(channel, 2));
    return CcuResult::CCU_SUCCESS;
}

void Phase(unsigned rank, const char* phase) {
    std::printf("PUBLIC_CCU_PHASE rank=%u phase=%s\n", rank, phase);
    std::fflush(stdout);
}

void SaveRoot(const std::string& path, const HcclRootInfo& info) {
    const std::string tmp = path + ".part";
    std::ofstream file(tmp, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&info), sizeof(info));
    file.close();
    if (!file || std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("publish HCCL bootstrap failed");
}

HcclRootInfo LoadRoot(const std::string& path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
        HcclRootInfo info{};
        std::ifstream file(path, std::ios::binary);
        if (file.read(reinterpret_cast<char*>(&info), sizeof(info))) return info;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    throw std::runtime_error("timeout waiting for bootstrap file");
}

void ValidationBarrier(const std::string& bootstrap, unsigned rank, unsigned iteration) {
    const auto prefix = bootstrap + ".validated." + std::to_string(iteration) + ".";
    { std::ofstream marker(prefix + std::to_string(rank)); marker << "validated\n";
      if (!marker) throw std::runtime_error("write validation barrier failed"); }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
        std::ifstream peer(prefix + std::to_string(1 - rank));
        std::string value;
        if (peer >> value && value == "validated") return;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("timeout waiting for peer validation");
}

struct SelectedPath { HcclChannelDesc desc{}; uint32_t die = 0; };
SelectedPath SelectDirect(HcclComm comm, uint32_t rank) {
    uint32_t *layers = nullptr, count = 0;
    CHECK(HcclRankGraphGetLayers(comm, &layers, &count));
    if (!layers || !count) throw std::runtime_error("empty rank graph");
    const std::vector<uint32_t> ownedLayers(layers, layers + count);
    for (auto layer : ownedLayers) {
        CommLink* links = nullptr;
        uint32_t num = 0;
        CHECK(HcclRankGraphGetLinks(comm, layer, rank, 1 - rank, &links, &num));
        for (uint32_t i = 0; i < num; ++i) {
            const auto link = links[i];
            if (link.linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP || link.linkAttr.hop != 1) continue;
            SelectedPath result;
            CHECK(HcclRankGraphGetEndpointInfo(comm, rank, &link.srcEndpointDesc,
                ENDPOINT_ATTR_DIE_ID, sizeof(result.die), &result.die));
            CHECK(HcclChannelDescInit(&result.desc, 1));
            result.desc.remoteRank = 1 - rank;
            result.desc.channelProtocol = link.linkAttr.linkProtocol;
            result.desc.localEndpoint = link.srcEndpointDesc;
            result.desc.remoteEndpoint = link.dstEndpointDesc;
            result.desc.notifyNum = 3;
            std::printf("PUBLIC_CCU_PATH rank=%u layer=%u hop=1 die=%u src_phy=%u dst_phy=%u\n",
                rank, layer, result.die, link.srcEndpointDesc.loc.device.devPhyId,
                link.dstEndpointDesc.loc.device.devPhyId);
            return result;
        }
    }
    throw std::runtime_error("no discovered one-hop UB_CTP link; no fabricated EID fallback");
}

size_t SendBytes(unsigned rank, unsigned iteration, size_t capacity) {
    switch (iteration % 4) {
        case 0: return capacity;
        case 1: return rank == 0 ? capacity : capacity / 2;
        case 2: return rank == 0 ? 0 : capacity;
        default: return 0;
    }
}
uint32_t Value(unsigned rank, unsigned iteration, size_t index) {
    return static_cast<uint32_t>((rank + 1) * 1000003U + iteration * 8191U + index);
}

void Library(const char* name, void* address) {
    Dl_info info{};
    if (!dladdr(address, &info) || !info.dli_fname) throw std::runtime_error("dladdr failed");
    std::printf("PUBLIC_CCU_LIBRARY symbol=%s path=%s\n", name, info.dli_fname);
}

void CheckAbi() {
    if (!HcommCcuKernelRegisterStart || !HcommCcuKernelRegister ||
        !HcommCcuKernelRegisterEnd || !HcommCcuKernelLaunch)
        throw std::runtime_error("public CCU weak symbol unavailable in loaded libhcomm");
    Library("HcommCcuKernelLaunch", reinterpret_cast<void*>(HcommCcuKernelLaunch));
    Library("HcclCommInitRootInfo", reinterpret_cast<void*>(HcclCommInitRootInfo));
}

void Run(unsigned rank, int device, const std::string& bootstrap, size_t capacity, unsigned iterations) {
    CheckAbi();
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(device));
    Phase(rank, "communicator_init");
    HcclRootInfo root{};
    if (rank == 0) { CHECK(HcclGetRootInfo(&root)); SaveRoot(bootstrap, root); }
    else root = LoadRoot(bootstrap);
    HcclComm comm = nullptr;
    CHECK(HcclCommInitRootInfo(2, &root, rank, &comm));

    Phase(rank, "query_instance");
    CcuInsHandle instance = 0;
    uint32_t instanceCount = 1;
    CHECK(HcclCommQueryCcuIns(comm, &instance, &instanceCount));
    if (!instance || instanceCount != 1) throw std::runtime_error("expected one communicator-owned CCU instance");
    // Query returns a borrowed instance: never Assign or Destroy this handle.
    auto path = SelectDirect(comm, rank);
    if (path.die > 1) throw std::runtime_error("unsupported local IO die");
    aclrtStream mainStream = nullptr, routeStream = nullptr;
    CHECK(aclrtCreateStream(&mainStream));
    ThreadHandle mainThread = 0, routeThread = 0;
    CHECK(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, mainStream, 3, &mainThread));
    routeStream = mainStream;
    routeThread = mainThread;
    if (path.die == 1) {
        CHECK(aclrtCreateStream(&routeStream));
        CHECK(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, routeStream, 3, &routeThread));
    }
    void *send = nullptr, *recv = nullptr;
    CHECK(aclrtMalloc(&send, capacity, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&recv, capacity, ACL_MEM_MALLOC_HUGE_FIRST));
    CommMem inputMem{COMM_MEM_TYPE_DEVICE, send, capacity};
    CommMem outputMem{COMM_MEM_TYPE_DEVICE, recv, capacity};
    HcclMemHandle memHandles[2]{};
    CHECK(HcclCommMemReg(comm, "public_ccu_send", &inputMem, &memHandles[0]));
    CHECK(HcclCommMemReg(comm, "public_ccu_recv", &outputMem, &memHandles[1]));
    path.desc.memHandles = memHandles;
    path.desc.memHandleNum = 2;
    Phase(rank, "channel_acquire");
    ChannelHandle channel = 0;
    CHECK(HcclChannelAcquire(comm, COMM_ENGINE_CCU, &path.desc, 1, &channel));
    uint64_t inputToken = 0, outputToken = 0;
    CHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(send), capacity, &inputToken));
    CHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(recv), capacity, &outputToken));

    Phase(rank, "kernel_register");
    const void* kernelArgs[] = {&channel};
    CcuKernelHandle kernel = 0;
    CHECK(HcommCcuKernelRegisterStart(instance));
    CHECK(HcommCcuKernelRegister(instance, path.die, "A5PublicApiDirectSmokeV1",
        reinterpret_cast<const void*>(ExchangeKernel), kernelArgs, 1, &kernel));
    CHECK(HcommCcuKernelRegisterEnd(instance));
    uint32_t taskArgsNum = 0;
    CHECK(HcommCcuGetTaskArgsNum(kernel, &taskArgsNum));
    if (taskArgsNum != 5) throw std::runtime_error("unexpected task argument count=" + std::to_string(taskArgsNum));
    std::vector<uint32_t> input(capacity / 4), output(capacity / 4);
    constexpr uint32_t sentinel = 0xfefefefeU;
    Phase(rank, "repeated_launch");
    for (unsigned it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < input.size(); ++i) input[i] = Value(rank, it, i);
        std::fill(output.begin(), output.end(), sentinel);
        CHECK(aclrtMemcpy(send, capacity, input.data(), capacity, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK(aclrtMemcpy(recv, capacity, output.data(), capacity, ACL_MEMCPY_HOST_TO_DEVICE));
        const uint64_t taskArgs[] = {reinterpret_cast<uint64_t>(send), reinterpret_cast<uint64_t>(recv),
            inputToken, outputToken, SendBytes(rank, it, capacity)};
        CHECK(HcommCcuKernelLaunch(routeThread, kernel, taskArgs, 5));
        CHECK(aclrtSynchronizeStream(routeStream));
        CHECK(aclrtMemcpy(output.data(), capacity, recv, capacity, ACL_MEMCPY_DEVICE_TO_HOST));
        const auto received = SendBytes(1 - rank, it, capacity) / 4;
        for (size_t i = 0; i < output.size(); ++i) {
            const auto expected = i < received ? Value(1 - rank, it, i) : sentinel;
            if (output[i] != expected) throw std::runtime_error("data/guard mismatch iteration=" +
                std::to_string(it) + " word=" + std::to_string(i));
        }
        // Out-of-band correctness barrier prevents the next write before both
        // previous results are checked. No native collective/kernel is inserted.
        ValidationBarrier(bootstrap, rank, it);
    }
    Phase(rank, "teardown");
    CHECK(HcclCommDestroy(comm)); // destroys owned instance/channel/memory registrations
    CHECK(aclrtFree(send));
    CHECK(aclrtFree(recv));
    if (routeStream != mainStream) CHECK(aclrtDestroyStream(routeStream));
    CHECK(aclrtDestroyStream(mainStream));
    CHECK(aclrtResetDevice(device));
    CHECK(aclFinalize());
    std::printf("PUBLIC_CCU_RANK_PASS rank=%u iterations=%u registrations=1 channels=1 capacity=%zu\n",
        rank, iterations, capacity);
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--check-abi") {
        try { CheckAbi(); std::puts("PUBLIC_CCU_ABI_PASS"); return 0; }
        catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
    }
    if (argc != 7) {
        std::fprintf(stderr, "usage: %s RANK LOGICAL_DEVICE BOOTSTRAP_FILE BYTES ITERATIONS CCU_SCHED\n", argv[0]);
        return 2;
    }
    unsigned rank = 0;
    try {
        rank = static_cast<unsigned>(std::stoul(argv[1]));
        const int device = std::stoi(argv[2]);
        const size_t bytes = std::stoull(argv[4]);
        const unsigned iterations = static_cast<unsigned>(std::stoul(argv[5]));
        if (rank > 1 || device < 0 || bytes < 64 || bytes > (1ULL << 30) || bytes % 64 || iterations < 4)
            throw std::runtime_error("invalid rank/device/bytes/iterations (bytes multiple of 64; iterations >=4)");
        if (std::string(argv[6]) != "CCU_SCHED") throw std::runtime_error("expected explicit CCU_SCHED mode");
        setenv("HCCL_OP_EXPANSION_MODE", "CCU_SCHED", 1);
        Run(rank, device, argv[3], bytes, iterations);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "PUBLIC_CCU_FAILED rank=%u error=%s\n", rank, error.what());
        std::fflush(nullptr);
        // Failed asynchronous launches must not be hidden by destructor cleanup.
        std::_Exit(1);
    }
}
