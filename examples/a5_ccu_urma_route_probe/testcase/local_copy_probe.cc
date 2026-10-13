// Isolated CCU local-copy diagnostic. No ChannelAcquire, remote WriteNb,
// ACL D2D memcpy, torch, or DeepEP. H2D/D2H are test setup/verification only.
#include "local_copy_probe_data.h"
#include "../op_kernel_ccu/grouped_local_copy_kernel.h"
#include <acl/acl.h>
#include <acl/acl_prof.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_res.h>
#include <hcomm/ccu/hccl_ccu_res.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

using namespace a5_local_probe;
namespace CcuRep = hcomm::CcuRep; // required by the target SDK's CCU_WHILE macro
namespace {
struct Options {
    uint32_t rank = 0, ranks = 2, changed = 3, queued = 20;
    size_t bytes = 4096;
    bool chunked = false, grouped = false, builder = false, benchmark = false;
    uint32_t warmup = 100, iterations = 20;
    uint32_t groupSliceBytes = 32768;
    std::string directory, profileDirectory;
};
Options options;
std::string phase;
void Mark(const char *value)
{
    phase = value;
    std::printf("LOCAL_COPY_PHASE rank=%u phase=%s\n", options.rank, value);
    std::fflush(stdout);
}
void Check(int status, const char *api)
{
    if (status) throw std::runtime_error(std::string(api) + " status=" + std::to_string(status));
}
uint64_t Uint(const char *text)
{
    if (!text || !*text || *text == '-') throw std::runtime_error("invalid unsigned argument");
    errno = 0; char *end = nullptr;
    auto value = std::strtoull(text, &end, 10);
    if (errno || *end) throw std::runtime_error("invalid unsigned argument");
    return value;
}
Options Parse(int argc, char **argv)
{
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--builder-check") { result.builder = true; continue; }
        if (key == "--benchmark") { result.benchmark = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("missing argument value");
        const char *value = argv[++i];
        if (key == "--case-dir") { result.directory = value; continue; }
        if (key == "--profile-dir") { result.profileDirectory = value; continue; }
        if (key == "--copy-mode") {
            if (std::string(value) != "direct" && std::string(value) != "chunked" && std::string(value) != "group")
                throw std::runtime_error("copy-mode must be direct, chunked or group");
            result.chunked = std::string(value) == "chunked";
            result.grouped = std::string(value) == "group"; continue;
        }
        const auto number = Uint(value);
        if (number > UINT32_MAX) throw std::runtime_error("argument exceeds uint32 range");
        if (key == "--rank") result.rank = static_cast<uint32_t>(number);
        else if (key == "--ranks") result.ranks = static_cast<uint32_t>(number);
        else if (key == "--bytes") result.bytes = static_cast<size_t>(number);
        else if (key == "--changed-iters") result.changed = static_cast<uint32_t>(number);
        else if (key == "--queued-iters") result.queued = static_cast<uint32_t>(number);
        else if (key == "--warmup") result.warmup = static_cast<uint32_t>(number);
        else if (key == "--iters") result.iterations = static_cast<uint32_t>(number);
        else if (key == "--group-slice-bytes") result.groupSliceBytes = static_cast<uint32_t>(number);
        else throw std::runtime_error("unknown argument: " + key);
    }
    if (!result.builder && (result.directory.empty() || result.ranks < 1 || result.ranks > 2 || result.rank >= result.ranks))
        throw std::runtime_error("hardware worker requires case-dir and valid rank/ranks (1 or 2)");
    if (result.changed < 1 || result.changed > 1000 || !ValidSize(result.bytes, result.queued, result.chunked))
        throw std::runtime_error("invalid size/count; chunked size must be a multiple of 4096; pinned budget <=256MiB/rank");
    if (result.warmup > 100000 || result.iterations < 1 || result.iterations > 100000 ||
        (!result.benchmark && !result.profileDirectory.empty()))
        throw std::runtime_error("invalid benchmark options; profiling requires --benchmark");
    if (!a5_ccu_urma_probe::GroupCopyConfig(result.groupSliceBytes).Valid() ||
        (!result.grouped && result.groupSliceBytes != 32768))
        throw std::runtime_error("group slice must be 32768/65536/131072 bytes; non-default requires group mode");
    return result;
}
const char *ModeName() { return options.grouped ? "group" : options.chunked ? "chunked" : "direct"; }
std::string GeometryJson()
{
    if (!options.grouped) return "";
    return ",\"group_slice_bytes\":" + std::to_string(options.groupSliceBytes) + ",\"group_lanes\":8";
}
class LocalArg : public hcomm::CcuKernelArg {
public:
    LocalArg(bool chunked,bool group,uint32_t sliceBytes) : chunked(chunked),group(group),config(sliceBytes) {}
    bool chunked,group;
    a5_ccu_urma_probe::GroupCopyConfig config;
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature signature;
        signature.Append(group ? "A5LocalGroup8xSliceV2:" + std::to_string(config.sliceBytes) :
            chunked ? "A5LocalCopyOnlyChunk4KV1" : "A5LocalCopyOnlyDirectV1");
        return signature;
    }
};
class LocalTask : public hcomm::CcuTaskArg {
public:
    uint64_t input, output, inputToken, outputToken, bytes;
    LocalTask(uint64_t i, uint64_t o, uint64_t it, uint64_t ot, uint64_t n)
        : input(i), output(o), inputToken(it), outputToken(ot), bytes(n) {}
};
class LocalKernel : public a5_ccu_urma_probe::GroupedLocalCopyKernel {
public:
    explicit LocalKernel(const hcomm::CcuKernelArg &arg) : GroupedLocalCopyKernel(arg)
    {
        const auto *local = dynamic_cast<const LocalArg *>(&arg);
        if (!local) throw std::runtime_error("wrong kernel argument type");
        chunked_ = local->chunked;
        grouped_ = local->group;
        config_ = local->config;
    }
protected:
    HcclResult Algorithm() override
    {
        auto input = CreateVariable(), output = CreateVariable();
        auto inputToken = CreateVariable(), outputToken = CreateVariable(), bytes = CreateVariable();
        Load(input); Load(output); Load(inputToken); Load(outputToken);
        auto source = CreateLocalAddr(), destination = CreateLocalAddr();
        source.addr = input; source.token = inputToken;
        destination.addr = output; destination.token = outputToken;
        if(grouped_) {
            std::array<hcomm::CcuRep::Variable,4> go;
            for(auto &v:go) {v=CreateVariable(); Load(v);}
            return GroupedLocalCopy(destination,source,go,config_);
        }
        Load(bytes);
        auto done = CreateCompletedEvent(); // one event, stable lifetime, reused only after Wait
        if (!chunked_) {
            const auto status = LocalCopyNb(destination, source, bytes, done);
            return status == HCCL_SUCCESS ? WaitEvent(done) : status;
        }
        auto remaining = CreateVariable(), chunk = CreateVariable(), decrement = CreateVariable();
        remaining = bytes; chunk = CHUNK;
        // This old object API has uint64 add, but no variable-variable compare
        // or subtract. Use modular uint64 decrement, with aligned sizes checked
        // before launch so the counter reaches exactly zero, never overshoots.
        decrement = 0ULL - static_cast<uint64_t>(CHUNK);
        CCU_WHILE(remaining != 0) {
            auto status = LocalCopyNb(destination, source, chunk, done);
            if (status != HCCL_SUCCESS) return status;
            status = WaitEvent(done);
            if (status != HCCL_SUCCESS) return status;
            source.addr += chunk; destination.addr += chunk; remaining += decrement;
        }
        return HCCL_SUCCESS;
    }
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override
    {
        const auto *task = dynamic_cast<const LocalTask *>(&arg);
        if (!task) return {};
        std::vector<uint64_t> values={task->input,task->output,task->inputToken,task->outputToken};
        if(grouped_) {
            std::array<uint64_t,4> go;
            if(!a5_ccu_urma_probe::BuildGroupCopyPlan(task->bytes,&go,config_)) return {};
            values.insert(values.end(),go.begin(),go.end());
        } else values.push_back(task->bytes);
        return values;
    }
private:
    bool chunked_ = false;
    bool grouped_ = false;
    a5_ccu_urma_probe::GroupCopyConfig config_;
};
std::unique_ptr<hcomm::CcuKernel> CreateKernel(const hcomm::CcuKernelArg &arg)
{
    return std::unique_ptr<hcomm::CcuKernel>(new LocalKernel(arg));
}
std::string Path(const std::string &name) { return options.directory + '/' + name; }
void PublishRoot(const HcclRootInfo &root)
{
    std::ofstream file(Path("root_info.tmp"), std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(&root), sizeof(root)); file.flush();
    if (!file.good()) throw std::runtime_error("root-info write failed");
    file.close();
    if (std::rename(Path("root_info.tmp").c_str(), Path("root_info.bin").c_str()))
        throw std::runtime_error("atomic root-info publish failed");
}
HcclRootInfo ReadRoot()
{
    HcclRootInfo root{};
    for (unsigned attempt = 0; attempt < 900; ++attempt) {
        std::ifstream file(Path("root_info.bin"), std::ios::binary);
        if (file.read(reinterpret_cast<char *>(&root), sizeof(root))) return root;
        std::ifstream failed(Path("rank0.status")); int status = 0;
        if ((failed >> status) && status) throw std::runtime_error("rank0 exited before root-info publication");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("root-info wait timed out");
}
void CleanupBarrier()
{
    std::ofstream done(Path("rank" + std::to_string(options.rank) + ".done"));
    done << "done\n"; done.close();
    for (unsigned attempt = 0; attempt < 900; ++attempt) {
        bool all = true;
        for (uint32_t rank = 0; rank < options.ranks; ++rank) {
            std::ifstream file(Path("rank" + std::to_string(rank) + ".done"));
            all &= file.good();
            std::ifstream failed(Path("rank" + std::to_string(rank) + ".status")); int status = 0;
            if ((failed >> status) && status) throw std::runtime_error("peer failed before cleanup barrier");
        }
        if (all) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("cleanup barrier timed out");
}
struct Resources {
    bool initialized = false;
    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    void *input = nullptr, *output = nullptr;
    uint8_t *hostInput = nullptr, *hostOutput = nullptr;
    // Only call with completed tasks. A fault/hang is stopped by the outer timeout;
    // do not free storage underneath unfinished CCU work or reset public hardware.
    void Close()
    {
        Mark("cleanup");
        if (comm) { Check(HcclCommDestroy(comm), "HcclCommDestroy"); comm = nullptr; }
        if (stream) { Check(aclrtDestroyStream(stream), "aclrtDestroyStream"); stream = nullptr; }
        if (input) { Check(aclrtFree(input), "aclrtFree(input)"); input = nullptr; }
        if (output) { Check(aclrtFree(output), "aclrtFree(output)"); output = nullptr; }
        if (hostInput) { Check(aclrtFreeHost(hostInput), "aclrtFreeHost(input)"); hostInput = nullptr; }
        if (hostOutput) { Check(aclrtFreeHost(hostOutput), "aclrtFreeHost(output)"); hostOutput = nullptr; }
        if (initialized) { Check(aclFinalize(), "aclFinalize"); initialized = false; }
    }
};
void VerifyBuffer(const uint8_t *buffer, uint32_t epoch, bool input)
{
    size_t bad = 0;
    if (!Verify(buffer, options.bytes, options.rank, epoch, input, &bad))
        throw std::runtime_error(std::string(input ? "input changed" : "output/guard mismatch") +
            " epoch=" + std::to_string(epoch) + " allocation_byte=" + std::to_string(bad));
}
void ShowLibrary(const char *api, void *symbol)
{
    if (!symbol) throw std::runtime_error(std::string("missing API ") + api);
    Dl_info info{};
    std::printf("LOCAL_COPY_LIBRARY api=%s path=%s\n", api,
        dladdr(symbol, &info) && info.dli_fname ? info.dli_fname : "unknown");
}
class Profile {
public:
    ~Profile() { // error-only best effort; never free pending device buffers
        if (started_) (void)aclprofStop(config_);
        if (config_) (void)aclprofDestroyConfig(config_);
        if (initialized_) (void)aclprofFinalize();
    }
    void Start()
    {
        if (options.profileDirectory.empty()) return;
        ShowLibrary("aclprofStart", reinterpret_cast<void *>(aclprofStart));
        Check(aclprofInit(options.profileDirectory.c_str(), options.profileDirectory.size()), "aclprofInit");
        initialized_ = true;
        int32_t current = -1;
        Check(aclrtGetDevice(&current), "aclrtGetDevice(profiler)");
        if (current < 0) throw std::runtime_error("invalid current device for profiler");
        uint32_t device = static_cast<uint32_t>(current);
        config_ = aclprofCreateConfig(&device, 1, ACL_AICORE_NONE, nullptr,
            ACL_PROF_TASK_TIME | ACL_PROF_ACL_API);
        if (!config_) throw std::runtime_error("aclprofCreateConfig returned null");
        std::printf("LOCAL_COPY_PROFILE rank=%u current_device=%u output=%s formal_calls=%u\n",
            options.rank, device, options.profileDirectory.c_str(), options.iterations);
        std::fflush(stdout);
        Check(aclprofStart(config_), "aclprofStart"); started_ = true;
    }
    void Stop()
    {
        if (!initialized_) return;
        Check(aclprofStop(config_), "aclprofStop"); started_ = false;
        Check(aclprofDestroyConfig(config_), "aclprofDestroyConfig"); config_ = nullptr;
        Check(aclprofFinalize(), "aclprofFinalize"); initialized_ = false;
    }
private:
    aclprofConfig *config_ = nullptr;
    bool started_ = false, initialized_ = false;
};

void Benchmark(Resources &r, ThreadHandle thread, CcuKernelHandle kernel, LocalTask &task)
{
    const size_t span = options.bytes + 2 * GUARD;
    const uint32_t epoch = options.changed + options.queued + 1;
    Mark("benchmark_setup");
    Fill(r.hostInput, options.bytes, options.rank, epoch, true);
    Fill(r.hostOutput, options.bytes, options.rank, epoch, false);
    Check(aclrtMemcpyAsync(r.input, span, r.hostInput, span, ACL_MEMCPY_HOST_TO_DEVICE, r.stream), "benchmark H2D input");
    Check(aclrtMemcpyAsync(r.output, span, r.hostOutput, span, ACL_MEMCPY_HOST_TO_DEVICE, r.stream), "benchmark H2D poison");
    Check(aclrtSynchronizeStream(r.stream), "benchmark setup synchronize");
    Mark("benchmark_warmup");
    for (uint32_t i = 0; i < options.warmup; ++i)
        Check(HcclCcuKernelLaunch(r.comm, thread, kernel, &task), "warmup local CCU launch");
    Check(aclrtSynchronizeStream(r.stream), "warmup batch synchronize");
    // Re-poison after warmup: a stale warmup output cannot pass formal checks.
    Check(aclrtMemcpyAsync(r.output, span, r.hostOutput, span, ACL_MEMCPY_HOST_TO_DEVICE, r.stream), "pre-profile poison");
    Check(aclrtSynchronizeStream(r.stream), "pre-profile synchronize");
    Profile profile;
    profile.Start();
    Mark("benchmark_measure");
    const auto begin = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < options.iterations; ++i)
        Check(HcclCcuKernelLaunch(r.comm, thread, kernel, &task), "formal local CCU launch");
    Check(aclrtSynchronizeStream(r.stream), "formal batch synchronize");
    const double totalUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
    profile.Stop();
    Mark("benchmark_readback");
    // All readback is AFTER profile.Stop; none of these copies are timed.
    Check(aclrtMemcpyAsync(r.hostOutput, span, r.output, span, ACL_MEMCPY_DEVICE_TO_HOST, r.stream), "post-profile output");
    Check(aclrtMemcpyAsync(r.hostInput, span, r.input, span, ACL_MEMCPY_DEVICE_TO_HOST, r.stream), "post-profile input");
    Check(aclrtSynchronizeStream(r.stream), "benchmark readback synchronize");
    VerifyBuffer(r.hostOutput, epoch, false); VerifyBuffer(r.hostInput, epoch, true);
    std::printf("LOCAL_COPY_PERF_JSON {\"rank\":%u,\"pid\":%d,\"copy_mode\":\"%s\",\"bytes\":%zu,"
        "\"warmup\":%u,\"iterations\":%u,\"host_batch_total_us\":%.9f,\"host_batch_avg_us\":%.9f,"
        "\"profile_requested\":%s,\"formal_h2d_d2h\":false,\"remote_write\":false,"
        "\"engine\":\"CCU\",\"post_check\":\"PASS\"%s}\n",
        options.rank, static_cast<int>(getpid()), ModeName(), options.bytes, options.warmup,
        options.iterations, totalUs, totalUs / options.iterations,
        options.profileDirectory.empty() ? "false" : "true", GeometryJson().c_str());
    std::fflush(stdout);
}
void Run()
{
    ShowLibrary("HcclCcuKernelLaunch", reinterpret_cast<void *>(HcclCcuKernelLaunch));
    ShowLibrary("HcclCommInitRootInfo", reinterpret_cast<void *>(HcclCommInitRootInfo));
    ShowLibrary("HcclGetRootInfo", reinterpret_cast<void *>(HcclGetRootInfo));
    ShowLibrary("HcclThreadAcquireWithStream", reinterpret_cast<void *>(HcclThreadAcquireWithStream));
    ShowLibrary("HcclCcuKernelRegister", reinterpret_cast<void *>(HcclCcuKernelRegister));
    ShowLibrary("HcclCcuKernelRegisterFinish", reinterpret_cast<void *>(HcclCcuKernelRegisterFinish));
    Resources resources;
    Mark("acl_init"); Check(aclInit(nullptr), "aclInit"); resources.initialized = true;
    Check(aclrtSetDevice(static_cast<int32_t>(options.rank)), "aclrtSetDevice");
    HcclRootInfo root{};
    Mark("root_info");
    if (options.rank == 0) { Check(HcclGetRootInfo(&root), "HcclGetRootInfo"); PublishRoot(root); }
    else root = ReadRoot();
    Mark("comm_init");
    Check(HcclCommInitRootInfo(options.ranks, &root, options.rank, &resources.comm), "HcclCommInitRootInfo");
    Check(aclrtCreateStream(&resources.stream), "aclrtCreateStream");
    ThreadHandle thread = 0;
    Mark("thread_acquire");
    Check(HcclThreadAcquireWithStream(resources.comm, COMM_ENGINE_CCU, resources.stream, 1, &thread),
          "HcclThreadAcquireWithStream");
    LocalArg arg(options.chunked,options.grouped,options.groupSliceBytes); // channels intentionally empty
    hcomm::KernelCreator creator = CreateKernel;
    CcuKernelHandle kernel = 0;
    Mark("kernel_register");
    Check(HcclCcuKernelRegister(resources.comm, &kernel, &creator, &arg), "HcclCcuKernelRegister");
    Check(HcclCcuKernelRegisterFinish(resources.comm), "HcclCcuKernelRegisterFinish");
    if (options.grouped) {
        const uint64_t ms=options.groupSliceBytes/4096;
        std::printf("LOCAL_COPY_GEOMETRY rank=%u lanes=8 slice_bytes=%u ms_stride=%lu total_ms=%lu\n",
            options.rank,options.groupSliceBytes,static_cast<unsigned long>(ms),static_cast<unsigned long>(8*ms));
    }
    std::printf("LOCAL_COPY_KERNEL_REGISTERED rank=%u mode=%s channels=0\n", options.rank,
                ModeName());
    const size_t span = options.bytes + 2 * GUARD, hostSpan = span * options.queued;
    Mark("allocate");
    Check(aclrtMalloc(&resources.input, span, ACL_MEM_MALLOC_HUGE_ONLY), "aclrtMalloc(input)");
    Check(aclrtMalloc(&resources.output, span, ACL_MEM_MALLOC_HUGE_ONLY), "aclrtMalloc(output)");
    Check(aclrtMallocHost(reinterpret_cast<void **>(&resources.hostInput), hostSpan), "aclrtMallocHost(input)");
    Check(aclrtMallocHost(reinterpret_cast<void **>(&resources.hostOutput), hostSpan), "aclrtMallocHost(output)");
    Mark("tokens");
    const auto inputToken = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(resources.input), span);
    const auto outputToken = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(resources.output), span);
    LocalTask task(reinterpret_cast<uint64_t>(resources.input) + GUARD,
        reinterpret_cast<uint64_t>(resources.output) + GUARD, inputToken, outputToken, options.bytes);
    auto enqueue = [&](uint8_t *hostInput, uint8_t *hostOutput) {
        Check(aclrtMemcpyAsync(resources.input, span, hostInput, span, ACL_MEMCPY_HOST_TO_DEVICE, resources.stream), "H2D input");
        Check(aclrtMemcpyAsync(resources.output, span, hostOutput, span, ACL_MEMCPY_HOST_TO_DEVICE, resources.stream), "H2D poison");
        Check(HcclCcuKernelLaunch(resources.comm, thread, kernel, &task), "HcclCcuKernelLaunch");
        Check(aclrtMemcpyAsync(hostOutput, span, resources.output, span, ACL_MEMCPY_DEVICE_TO_HOST, resources.stream), "D2H snapshot");
        // Each pinned input slice is read by H2D first, then overwritten by
        // its D2H snapshot on the same stream. It is never reused by another
        // queued invocation, so source preservation is checked for every call.
        Check(aclrtMemcpyAsync(hostInput, span, resources.input, span, ACL_MEMCPY_DEVICE_TO_HOST, resources.stream), "D2H input snapshot");
    };
    Mark("changed_payload");
    for (uint32_t epoch = 0; epoch < options.changed; ++epoch) {
        Fill(resources.hostInput, options.bytes, options.rank, epoch, true);
        Fill(resources.hostOutput, options.bytes, options.rank, epoch, false);
        enqueue(resources.hostInput, resources.hostOutput);
        Check(aclrtSynchronizeStream(resources.stream), "changed-payload synchronize");
        VerifyBuffer(resources.hostOutput, epoch, false); VerifyBuffer(resources.hostInput, epoch, true);
    }
    Mark("queued_payload");
    for (uint32_t index = 0; index < options.queued; ++index) {
        const uint32_t epoch = options.changed + index;
        Fill(resources.hostInput + index * span, options.bytes, options.rank, epoch, true);
        Fill(resources.hostOutput + index * span, options.bytes, options.rank, epoch, false);
    }
    for (uint32_t index = 0; index < options.queued; ++index)
        enqueue(resources.hostInput + index * span, resources.hostOutput + index * span);
    // No host synchronization between queued invocations; inspect every snapshot.
    Check(aclrtSynchronizeStream(resources.stream), "queued-payload synchronize");
    for (uint32_t index = 0; index < options.queued; ++index) {
        VerifyBuffer(resources.hostOutput + index * span, options.changed + index, false);
        VerifyBuffer(resources.hostInput + index * span, options.changed + index, true);
    }
    if (options.benchmark) Benchmark(resources, thread, kernel, task);
    std::printf("LOCAL_COPY_RESULT_JSON {\"rank\":%u,\"copy_mode\":\"%s\",\"bytes\":%zu,"
        "\"changed_calls\":%u,\"queued_calls\":%u,\"guards\":\"PASS\",\"status\":\"PASS\","
        "\"engine\":\"CCU\",\"channel_count\":0,\"acl_d2d\":false%s}\n",
        options.rank, ModeName(), options.bytes, options.changed, options.queued, GeometryJson().c_str());
    std::fflush(stdout);
    Mark("cleanup_barrier"); CleanupBarrier(); resources.Close();
    std::printf("LOCAL_COPY_WORKER_PASS rank=%u LocalCopyNb+WaitEvent data+guards+cleanup=PASS\n", options.rank);
}
} // namespace

int main(int argc, char **argv)
{
    try {
        options = Parse(argc, argv);
        if (options.builder) {
            LocalArg arg(options.chunked,options.grouped,options.groupSliceBytes); LocalKernel kernel(arg);
            Check(kernel.Init(), "CcuKernel::Init");
            std::printf("LOCAL_COPY_BUILDER_PASS mode=%s slice_bytes=%u graph_only=1 hardware_copy_executed=0\n",
                ModeName(),options.grouped ? options.groupSliceBytes : 0);
        } else Run();
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "LOCAL_COPY_FAILURE rank=%u phase=%s error=%s\n", options.rank, phase.c_str(), error.what());
        // A device error must not lead to freeing buffers below pending work.
        // Process exit/driver reclaims this isolated test's resources; no reset.
        return 1;
    }
}
