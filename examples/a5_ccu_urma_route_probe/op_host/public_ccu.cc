#include "../common/public_ccu.h"
#include <atomic>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>

namespace a5ccu {
namespace {
struct Registered {
    std::unique_ptr<CcuKernel> kernel;
    HcclComm owner = nullptr;
    int32_t device = -1;
};
std::mutex mutex;
std::map<std::pair<int32_t, uint64_t>, std::shared_ptr<Registered>> kernels;
std::set<std::pair<int32_t, HcclComm>> poisoned;
std::atomic<uint64_t> sequence{0};
CcuResult BuildKernel(::CcuKernelArg arg) {
    try { return static_cast<CcuResult>(static_cast<CcuKernel*>(arg)->Build()); }
    catch (const AscendC::ccu::detail::CcuException& error) {
        std::fprintf(stderr, "PUBLIC_CCU_BUILD_FAILED error=%s\n", error.what());
        return error.code();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "PUBLIC_CCU_BUILD_FAILED error=%s\n", error.what());
        return CcuResult::CCU_E_INTERNAL;
    }
}
}
HcclResult RegisterKernel(HcclComm comm, uint32_t die, uint64_t* handle,
                          const KernelCreator* creator, const CcuKernelArg* arg) {
    if (!comm || !handle || !creator || !*creator || !arg) return HCCL_E_PTR;
    *handle = 0;
    if (die > 1) return HCCL_E_PARA;
    if (!HcommCcuKernelRegisterStart || !HcommCcuKernelRegister ||
        !HcommCcuKernelRegisterEnd || !HcommCcuKernelLaunch) return HCCL_E_NOT_SUPPORT;
    std::lock_guard<std::mutex> lock(mutex);
    int32_t device = -1;
    if (aclrtGetDevice(&device) != ACL_SUCCESS) return HCCL_E_RUNTIME;
    const auto ownerKey = std::make_pair(device, comm);
    if (poisoned.count(ownerKey)) return HCCL_E_UNAVAIL;
    CcuInsHandle instance = 0;
    uint32_t count = 1;
    auto status = HcclCommQueryCcuIns(comm, &instance, &count);
    if (status != HCCL_SUCCESS) return status;
    if (count != 1 || !instance) return HCCL_E_UNAVAIL;
    // Borrow the communicator-owned instance. Never Assign or Destroy it.
    auto entry = std::make_shared<Registered>();
    entry->owner = comm; entry->device = device;
    try { entry->kernel = (*creator)(*arg); }
    catch (...) { return HCCL_E_INTERNAL; }
    if (!entry->kernel) return HCCL_E_PTR;
    const void* args[] = {entry->kernel.get()};
    CcuKernelHandle registered = 0;
    const std::string name = "a5_public_" + std::to_string(++sequence);
    auto ccuStatus = HcommCcuKernelRegisterStart(instance);
    if (ccuStatus != CcuResult::CCU_SUCCESS) return Status(ccuStatus);
    // Once a session starts, any failure poisons this communicator for this
    // library. Never retry a partially built instruction/resource session.
    ccuStatus = HcommCcuKernelRegister(instance, die, name.c_str(),
        reinterpret_cast<const void*>(BuildKernel), args, 1, &registered);
    if (ccuStatus == CcuResult::CCU_SUCCESS) ccuStatus = HcommCcuKernelRegisterEnd(instance);
    uint32_t actualArgs = 0;
    if (ccuStatus == CcuResult::CCU_SUCCESS) ccuStatus = HcommCcuGetTaskArgsNum(registered, &actualArgs);
    if (ccuStatus != CcuResult::CCU_SUCCESS || actualArgs != entry->kernel->ArgumentCount()) {
        poisoned.insert(ownerKey);
        return ccuStatus != CcuResult::CCU_SUCCESS ? Status(ccuStatus) : HCCL_E_PARA;
    }
    const auto key = std::make_pair(device, static_cast<uint64_t>(registered));
    if (!kernels.emplace(key, entry).second) { poisoned.insert(ownerKey); return HCCL_E_PARA; }
    *handle = registered;
    std::printf("PUBLIC_CCU_REGISTER comm=%p die=%u task_args=%u kernel=%llu\n", comm, die,
        actualArgs, static_cast<unsigned long long>(registered));
    return HCCL_SUCCESS;
}
HcclResult LaunchKernel(HcclComm comm, ThreadHandle thread, uint64_t handle, const CcuTaskArg* task) {
    if (!comm || !task) return HCCL_E_PTR;
    int32_t device = -1;
    if (aclrtGetDevice(&device) != ACL_SUCCESS) return HCCL_E_RUNTIME;
    std::shared_ptr<Registered> entry;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto found = kernels.find({device, handle});
        if (found == kernels.end()) return HCCL_E_NOT_FOUND;
        entry = found->second;
    }
    if (entry->owner != comm) return HCCL_E_PARA;
    const auto args = entry->kernel->Args(*task);
    if (args.empty() || args.size() != entry->kernel->ArgumentCount()) return HCCL_E_PARA;
    // No instance query, registration, channel creation or host synchronization.
    return Status(HcommCcuKernelLaunch(thread, handle, args.data(), static_cast<uint32_t>(args.size())));
}
} // namespace a5ccu
extern "C" int A5CcuPublicApiAbiVersion() { return 1; }
