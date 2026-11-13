// Two communication nodes in ONE capture; direct + explicit relay catalog.
// Reuse the baseline's bootstrap, validation barrier and error helpers.
#define main A5DirectBaselineMain
#include "../../a5_ccu_public_api_smoke/main.cc"
#undef main
#include "../op_host/utils.h"
#include <atomic>
#include <array>
#include "graph_gate_scheduler.h"

namespace {
struct Catalog { std::vector<ChannelHandle> channels; };
CcuResult PolicyKernel(CcuKernelArg arg)
{
    const auto &channels = static_cast<const Catalog *>(arg)->channels;
    ccu::Variable input, output, inputToken, outputToken;
    EMIT(ccu::LoadArg(input, 0)); EMIT(ccu::LoadArg(output, 1));
    EMIT(ccu::LoadArg(inputToken, 2)); EMIT(ccu::LoadArg(outputToken, 3));
    std::vector<ccu::Variable> outputs, tokens, sizes;
    std::vector<ccu::Event> events(channels.size());
    for (size_t i = 0; i < channels.size(); ++i) {
        const auto channel = channels[i];
        outputs.push_back(ccu::GetResByChannel<ccu::Variable>(channel, 0));
        tokens.push_back(ccu::GetResByChannel<ccu::Variable>(channel, 1));
        EMIT(ccu::WriteVariableWithNotify(channel, output, 0, 0));
        EMIT(ccu::WriteVariableWithNotify(channel, outputToken, 1, 1));
        sizes.emplace_back();
        // CANN 9.3 RegisterEnd rejects derived variable addresses used by
        // Load/Store in this kernel. Pass each address directly from the host;
        // retain runtime loads so graph replay never freezes policy VALUES.
        ccu::Variable policyAddr, auditAddr;
        EMIT(ccu::LoadArg(policyAddr, static_cast<uint32_t>(4 + i)));
        EMIT(ccu::LoadArg(auditAddr, static_cast<uint32_t>(4 + channels.size() + i)));
        EMIT(ccu::Load(policyAddr, sizes.back()));
        EMIT(ccu::Store(auditAddr, sizes.back())); // audit actual device-side policy
    }
    for (auto channel : channels) {
        EMIT(ccu::NotifyWait(channel, 0)); EMIT(ccu::NotifyWait(channel, 1));
    }
    // Offset is a prefix sum of host-published byte counts, loaded AFTER gate.
    for (size_t i = 0; i < channels.size(); ++i) {
        ccu::LocalAddr src; src.addr = input; src.token = inputToken;
        ccu::RemoteAddr dst; dst.addr = outputs[i]; dst.token = tokens[i];
        for (size_t j = 0; j < i; ++j) { src.addr += sizes[j]; dst.addr += sizes[j]; }
        CCU_IF(sizes[i] != 0) { EMIT(ccu::Write(channels[i], dst, src, sizes[i], events[i])); }
    }
    for (size_t i = 0; i < channels.size(); ++i) {
        CCU_IF(sizes[i] != 0) { EMIT(ccu::EventWait(events[i])); }
    }
    EMIT(ccu::NotifyRecord(channels.front(), 2));
    EMIT(ccu::NotifyWait(channels.front(), 2));
    return CcuResult::CCU_SUCCESS;
}

struct Notice {
    std::atomic<unsigned> count{0};
    uint64_t *pinned = nullptr;
    uint64_t *device = nullptr;
};
void Callback(void *p) { static_cast<Notice *>(p)->count.fetch_add(1, std::memory_order_release); }
bool Seen(const Notice &n, bool callback)
{
    // Pinned D2H visibility is part of the target-machine validation, not a
    // C++ inter-thread atomic protocol. Runtime owns the DMA write.
    return callback ? n.count.load(std::memory_order_acquire) == 1 :
        *static_cast<volatile uint64_t *>(n.pinned) == 1;
}
void EnqueueNotice(Notice &n, bool callback, aclrtStream stream)
{
    if (callback) { CHECK(aclrtLaunchHostFunc(stream, Callback, &n)); return; }
    CHECK(aclrtValueWrite(n.device, 1, 0, stream));
    CHECK(aclrtMemcpyAsync(n.pinned, 8, n.device, 8, ACL_MEMCPY_DEVICE_TO_HOST, stream));
}
void Await(Notice &n, bool callback)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!Seen(n, callback)) {
        if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("notification timeout");
        std::this_thread::sleep_for(std::chrono::microseconds(GatePollUs()));
    }
}
struct Node {
    uint64_t *gate = nullptr, *policy = nullptr, *audit = nullptr, *started = nullptr;
    void *send = nullptr, *recv = nullptr;
    uint64_t inputToken = 0, outputToken = 0;
    Notice arrival, done;
};
void Alloc(void **p, size_t bytes) { CHECK(aclrtMalloc(p, bytes, ACL_MEM_MALLOC_HUGE_FIRST)); }
void AllocWord(uint64_t **p) { Alloc(reinterpret_cast<void **>(p), 8); }

// Isolate graph notification support from HCCL/channel/relay provisioning.
int NotificationProbe(int device, const std::string &kind)
{
    if (device < 0 || (kind != "CALLBACK" && kind != "D2H")) throw std::runtime_error("invalid probe args");
    const bool callback = kind == "CALLBACK";
    CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(device));
    aclrtStream exec = nullptr, control = nullptr;
    CHECK(aclrtCreateStream(&exec)); CHECK(aclrtCreateStream(&control));
    std::array<Notice, 4> notices;
    std::array<uint64_t *, 2> gates{};
    for (auto &n : notices) {
        AllocWord(&n.device); CHECK(aclrtMallocHost(reinterpret_cast<void **>(&n.pinned), 8));
        *n.pinned = 0; CHECK(aclrtValueWrite(n.device, 0, 0, control));
    }
    for (auto &gate : gates) { AllocWord(&gate); CHECK(aclrtValueWrite(gate, 0, 0, control)); }
    CHECK(aclrtSynchronizeStream(control));
    aclmdlRI model = nullptr;
    CHECK(aclmdlRICaptureBegin(exec, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL));
    for (unsigned op = 0; op < 2; ++op) {
        EnqueueNotice(notices[2 * op], callback, exec);
        CHECK(aclrtValueWait(gates[op], 1, ACL_STREAM_WAIT_VALUE_EQ, exec));
        EnqueueNotice(notices[2 * op + 1], callback, exec);
        CHECK(aclrtValueWrite(gates[op], 0, 0, exec));
    }
    CHECK(aclmdlRICaptureEnd(exec, &model));
    for (auto &n : notices) if (Seen(n, callback)) throw std::runtime_error("notification ran during capture");
    std::printf("NOTIFICATION_CAPTURE_PASS notify=%s\n", kind.c_str()); std::fflush(stdout);
    for (unsigned it = 0; it < 3; ++it) {
        for (auto &n : notices) { n.count = 0; *n.pinned = 0; }
        CHECK(aclmdlRIExecuteAsync(model, exec));
        for (unsigned op = 0; op < 2; ++op) {
            Await(notices[2 * op], callback);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (Seen(notices[2 * op + 1], callback) || (op == 0 && Seen(notices[2], callback)))
                throw std::runtime_error("notification gate ordering failure");
            CHECK(aclrtValueWrite(gates[op], 1, 0, control));
            Await(notices[2 * op + 1], callback);
        }
        CHECK(aclrtSynchronizeStream(exec)); CHECK(aclrtSynchronizeStream(control));
        if (callback) for (auto &n : notices) if (n.count != 1) throw std::runtime_error("callback accounting");
    }
    CHECK(aclmdlRIDestroy(model));
    for (auto &n : notices) { CHECK(aclrtFree(n.device)); CHECK(aclrtFreeHost(n.pinned)); }
    for (auto gate : gates) CHECK(aclrtFree(gate));
    CHECK(aclrtDestroyStream(exec)); CHECK(aclrtDestroyStream(control));
    CHECK(aclrtResetDevice(device)); CHECK(aclFinalize());
    std::printf("NOTIFICATION_REPLAY_PASS notify=%s nodes=2 iterations=3\n", kind.c_str());
    return 0;
}

int GateMain(int argc, char **argv)
{
    if (argc != 10) throw std::runtime_error(
        "usage: graph_gate RANK DEVICE BOOTSTRAP BYTES ITERS MANIFEST CALLBACK|D2H GRAPH|EAGER HOLD_MS");
    const unsigned rank = std::stoul(argv[1]), iterations = std::stoul(argv[5]);
    const int device = std::stoi(argv[2]);
    const std::string bootstrap = argv[3], manifest = argv[6], notify = argv[7], mode = argv[8];
    const size_t bytes = std::stoull(argv[4]);
    const unsigned hold = std::stoul(argv[9]);
    if (rank > 1 || device < 0 || bytes < 1024 || bytes > (1U << 28) || bytes % 1024 ||
        iterations < 4 || hold > 10000 || (notify != "CALLBACK" && notify != "D2H") ||
        (mode != "GRAPH" && mode != "EAGER")) throw std::runtime_error("invalid arguments");
    const bool callback = notify == "CALLBACK", graph = mode == "GRAPH";
    setenv("HCCL_OP_EXPANSION_MODE", "CCU_SCHED", 1);
    CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(device));
    HcclRootInfo root{};
    if (rank == 0) { CHECK(HcclGetRootInfo(&root)); SaveRoot(bootstrap, root); }
    else root = LoadRoot(bootstrap);
    HcclComm comm = nullptr; CHECK(HcclCommInitRootInfo(2, &root, rank, &comm));
    aclrtStream main = nullptr, control = nullptr, observe = nullptr;
    CHECK(aclrtCreateStream(&main)); CHECK(aclrtCreateStream(&control)); CHECK(aclrtCreateStream(&observe));
    a5_ccu_urma_probe::RoutePlanRequest plan;
    const bool relayOnly = std::getenv("A5_GATE_RELAY_ONLY") && std::string(std::getenv("A5_GATE_RELAY_ONLY")) == "1";
    plan.includeDiscoveredRoute = !relayOnly; plan.relayManifest = manifest;
    if (!relayOnly) plan.weights = {1};
    // Count manifest rows; the production parser validates schema/EIDs/duplicates.
    { std::ifstream f(manifest); std::string line;
      if (!std::getline(f, line)) throw std::runtime_error("missing relay manifest");
      while (std::getline(f, line)) if (!line.empty() && line[0] != '#') plan.weights.push_back(1); }
    const size_t relayCount = plan.weights.size() - (relayOnly ? 0 : 1);
    if (relayCount < 1 || relayCount > 2) throw std::runtime_error("requires 1 or 2 explicit relays");
    a5_ccu_urma_probe::RouteResources resources;
    CHECK(a5_ccu_urma_probe::GetRouteResources(comm, main,
        a5_ccu_urma_probe::RouteKernelKind::ROUTE_WRITE, &resources, &plan));
    aclrtStream exec = resources.slaveStream ? resources.slaveStream : main;
    Catalog catalog{resources.channels};
    const size_t paths = catalog.channels.size();
    if (paths != plan.weights.size()) throw std::runtime_error("catalog mismatch");
    const char *fixedPolicy = std::getenv("A5_GATE_POLICY");
    const int fixed = fixedPolicy ? std::stoi(fixedPolicy) : -1;
    if (fixed < -1 || fixed > static_cast<int>(paths)) throw std::runtime_error("invalid fixed policy");
    CcuInsHandle instance = 0; uint32_t count = 1;
    CHECK(HcclCommQueryCcuIns(comm, &instance, &count));
    const void *args[] = {&catalog}; CcuKernelHandle kernel = 0;
    CHECK(HcommCcuKernelRegisterStart(instance));
    CHECK(HcommCcuKernelRegister(instance, resources.dieId, "A5GraphGateRelayV2HostAddresses",
        reinterpret_cast<const void *>(PolicyKernel), args, 1, &kernel));
    CHECK(HcommCcuKernelRegisterEnd(instance));
    uint32_t argCount = 0; CHECK(HcommCcuGetTaskArgsNum(kernel, &argCount));
    if (argCount != 4 + 2 * paths) throw std::runtime_error("task ABI mismatch");
    std::array<Node, 2> nodes;
    std::vector<HcclMemHandle> registrations;
    for (auto &n : nodes) {
        Alloc(&n.send, bytes); Alloc(&n.recv, bytes);
        AllocWord(&n.gate); AllocWord(&n.started);
        Alloc(reinterpret_cast<void **>(&n.policy), paths * 8);
        Alloc(reinterpret_cast<void **>(&n.audit), paths * 8);
        for (auto *notice : {&n.arrival, &n.done}) {
            AllocWord(&notice->device);
            CHECK(aclrtMallocHost(reinterpret_cast<void **>(&notice->pinned), 8));
        }
        for (void *buffer : {n.send, n.recv}) {
            CommMem mem{COMM_MEM_TYPE_DEVICE, buffer, bytes}; HcclMemHandle handle{};
            const auto name = std::string("gate_buffer_") + std::to_string(registrations.size());
            CHECK(HcclCommMemReg(comm, name.c_str(), &mem, &handle)); registrations.push_back(handle);
        }
        CHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(n.send), bytes, &n.inputToken));
        CHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(n.recv), bytes, &n.outputToken));
    }
    auto submit = [&] {
        for (unsigned op = 0; op < 2; ++op) {
            auto &n = nodes[op];
            EnqueueNotice(n.arrival, callback, exec); // INSIDE capture, per operator
            CHECK(aclrtValueWait(n.gate, 1, ACL_STREAM_WAIT_VALUE_EQ, exec));
            CHECK(aclrtValueWrite(n.started, 1, 0, exec));
            std::vector<uint64_t> task = {reinterpret_cast<uint64_t>(n.send),
                reinterpret_cast<uint64_t>(n.recv), n.inputToken, n.outputToken};
            for (size_t p = 0; p < paths; ++p)
                task.push_back(reinterpret_cast<uint64_t>(n.policy + p));
            for (size_t p = 0; p < paths; ++p)
                task.push_back(reinterpret_cast<uint64_t>(n.audit + p));
            CHECK(HcommCcuKernelLaunch(resources.routeThread, kernel, task.data(),
                                      static_cast<uint32_t>(task.size())));
            EnqueueNotice(n.done, callback, exec);
            CHECK(aclrtValueWrite(n.gate, 0, 0, exec));
            if (op == 0) CHECK(aclrtMemcpyAsync(nodes[1].send, bytes, n.recv, bytes,
                                               ACL_MEMCPY_DEVICE_TO_DEVICE, exec));
        }
    };
    aclmdlRI model = nullptr;
    if (graph) {
        CHECK(aclmdlRICaptureBegin(exec, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL));
        // Any rejection is fatal and names the failing API. No eager fallback.
        submit(); CHECK(aclmdlRICaptureEnd(exec, &model));
        for (auto &n : nodes) if (n.arrival.count.load() || n.done.count.load())
            throw std::runtime_error("callback ran during capture");
        Phase(rank, "GRAPH_CAPTURE_PASS");
    }
    std::vector<uint32_t> input(bytes / 4), output(bytes / 4);
    uint64_t *hostStarted = nullptr; CHECK(aclrtMallocHost(reinterpret_cast<void **>(&hostStarted), 8));
    GateSchedulerClient scheduler(rank);
    const char *delayEnv = std::getenv("A5_GATE_GRANT_DELAY_MS");
    const unsigned grantDelay = delayEnv ? std::stoul(delayEnv) : 0;
    if (grantDelay > 10000) throw std::runtime_error("invalid grant delay");
    for (unsigned it = 0; it < iterations; ++it) {
        for (size_t i = 0; i < input.size(); ++i) input[i] = Value(rank, it, i);
        CHECK(aclrtMemcpy(nodes[0].send, bytes, input.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        for (auto &n : nodes) {
            n.arrival.count = 0; n.done.count = 0; *n.arrival.pinned = 0; *n.done.pinned = 0;
            CHECK(aclrtMemset(n.recv, bytes, 0xfe, bytes));
            CHECK(aclrtMemset(n.audit, paths * 8, 0, paths * 8));
            for (auto word : {n.gate, n.started, n.arrival.device, n.done.device})
                CHECK(aclrtValueWrite(word, 0, 0, control));
        }
        CHECK(aclrtSynchronizeStream(control));
        ValidationBarrier(bootstrap + ".armed", rank, it);
        if (graph) CHECK(aclmdlRIExecuteAsync(model, exec)); else submit();
        for (unsigned op = 0; op < 2; ++op) {
            auto &n = nodes[op]; Await(n.arrival, callback);
            std::printf("GATE_ARRIVED rank=%u iteration=%u op=%u\n", rank, it, op); std::fflush(stdout);
            if (op == 0 && Seen(nodes[1].arrival, callback)) throw std::runtime_error("node2 arrived before node1 release");
            std::this_thread::sleep_for(std::chrono::milliseconds(hold));
            CHECK(aclrtMemcpyAsync(hostStarted, 8, n.started, 8, ACL_MEMCPY_DEVICE_TO_HOST, observe));
            CHECK(aclrtSynchronizeStream(observe));
            if (*hostStarted || Seen(n.done, callback)) throw std::runtime_error("gate bypassed");
            std::vector<uint64_t> split(paths, 0);
            if (scheduler.Enabled()) {
                // May wait for another collective's ALL-RANK DONE. No ACL calls
                // in the callback; the device remains parked at its graph gate.
                split = scheduler.Arrive(it, op, paths, bytes);
                CHECK(aclrtMemcpyAsync(hostStarted, 8, n.started, 8, ACL_MEMCPY_DEVICE_TO_HOST, observe));
                CHECK(aclrtSynchronizeStream(observe));
                if (*hostStarted || Seen(n.done, callback)) throw std::runtime_error("gate bypassed while awaiting scheduler");
                std::printf("SCHED_GRANTED rank=%u iteration=%u op=%u ticket=%lu\n", rank, it, op,
                            static_cast<unsigned long>(scheduler.Ticket())); std::fflush(stdout);
                // Debug contention window only; default 0 in performance runs.
                std::this_thread::sleep_for(std::chrono::milliseconds(grantDelay));
            } else {
                const unsigned selected = fixed >= 0 ? static_cast<unsigned>(fixed) : (it + op) % (paths + 1);
                if (selected < paths) split[selected] = bytes;
                else if (relayOnly) { for (auto &value : split) value = bytes / paths; }
                else { split[0] = bytes / 2; for (size_t p = 1; p < paths; ++p) split[p] = bytes / (2 * (paths - 1)); }
            }
            for (size_t p = 0; p < paths; ++p) CHECK(aclrtValueWrite(n.policy + p, split[p], 0, control));
            const auto releaseAt = std::chrono::steady_clock::now();
            CHECK(aclrtValueWrite(n.gate, 1, 0, control)); // publish payload BEFORE READY
            Await(n.done, callback);
            const auto elapsedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - releaseAt).count();
            if (scheduler.Enabled()) scheduler.Done(it, op, static_cast<uint64_t>(elapsedNs));
            std::vector<uint64_t> audit(paths);
            CHECK(aclrtMemcpyAsync(audit.data(), paths * 8, n.audit, paths * 8, ACL_MEMCPY_DEVICE_TO_HOST, observe));
            CHECK(aclrtMemcpyAsync(output.data(), bytes, n.recv, bytes, ACL_MEMCPY_DEVICE_TO_HOST, observe));
            CHECK(aclrtSynchronizeStream(observe));
            if (audit != split) throw std::runtime_error("device policy audit mismatch");
            for (size_t i = 0; i < output.size(); ++i)
                if (output[i] != Value(op == 0 ? 1 - rank : rank, it, i)) throw std::runtime_error("payload mismatch");
            ValidationBarrier(bootstrap + ".op" + std::to_string(op), rank, it);
            std::printf("GATE_OP_PASS rank=%u iteration=%u op=%u mode=%s notify=%s bytes=", rank, it, op, mode.c_str(), notify.c_str());
            for (auto b : audit) std::printf("%lu,", static_cast<unsigned long>(b));
            std::puts(""); std::fflush(stdout);
        }
        CHECK(aclrtSynchronizeStream(exec)); CHECK(aclrtSynchronizeStream(control));
        if (callback) for (auto &n : nodes) if (n.arrival.count != 1 || n.done.count != 1)
            throw std::runtime_error("duplicate/lost callback");
    }
    if (graph) CHECK(aclmdlRIDestroy(model));
    CHECK(HcclCommDestroy(comm));
    for (auto &n : nodes) {
        for (void *p : {n.send, n.recv, static_cast<void *>(n.gate), static_cast<void *>(n.started),
                       static_cast<void *>(n.policy), static_cast<void *>(n.audit)}) CHECK(aclrtFree(p));
        for (auto *notice : {&n.arrival, &n.done}) { CHECK(aclrtFree(notice->device)); CHECK(aclrtFreeHost(notice->pinned)); }
    }
    CHECK(aclrtFreeHost(hostStarted));
    if (exec != main) CHECK(aclrtDestroyStream(exec));
    CHECK(aclrtDestroyStream(main)); CHECK(aclrtDestroyStream(control)); CHECK(aclrtDestroyStream(observe));
    CHECK(aclrtResetDevice(device)); CHECK(aclFinalize());
    std::printf("GRAPH_GATE_RANK_PASS rank=%u iterations=%u paths=%zu\n", rank, iterations, paths);
    return 0;
}
} // namespace
int main(int argc, char **argv)
{
    try {
        if (argc == 4 && std::string(argv[1]) == "--notification-probe")
            return NotificationProbe(std::stoi(argv[2]), argv[3]);
        if (argc == 2 && std::string(argv[1]) == "--check-abi") {
            CheckAbi();
            Library("aclrtValueWait", reinterpret_cast<void *>(aclrtValueWait));
            Library("aclrtValueWrite", reinterpret_cast<void *>(aclrtValueWrite));
            Library("aclrtLaunchHostFunc", reinterpret_cast<void *>(aclrtLaunchHostFunc));
            Library("aclmdlRICaptureBegin", reinterpret_cast<void *>(aclmdlRICaptureBegin));
            Library("aclmdlRIExecuteAsync", reinterpret_cast<void *>(aclmdlRIExecuteAsync));
            std::puts("GRAPH_GATE_ABI_PASS (compile/link/load only)"); return 0;
        }
        return GateMain(argc, argv);
    }
    catch (const std::exception &e) { std::fprintf(stderr, "GRAPH_GATE_FAIL %s\n", e.what()); std::fflush(nullptr); std::_Exit(1); }
}
