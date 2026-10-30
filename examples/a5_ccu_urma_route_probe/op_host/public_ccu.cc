#include "../common/public_ccu.h"
#include "../common/create_trace.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <unistd.h>

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
    try {
        auto* kernel=static_cast<CcuKernel*>(arg);
        const auto status=static_cast<CcuResult>(kernel->Build());
        const char* diagnostic=std::getenv("A5_CCU_RESOURCE_DIAGNOSTICS");
        if (status!=CcuResult::CCU_SUCCESS || (diagnostic && std::strcmp(diagnostic,"1")==0))
            std::fprintf(stderr,"PUBLIC_CCU_BUILD_STATUS status=%d task_args=%u\n",int(status),kernel->ArgumentCount());
        return status;
    }
    catch (const AscendC::ccu::detail::CcuException& error) {
        std::fprintf(stderr, "PUBLIC_CCU_BUILD_FAILED error=%s\n", error.what());
        return error.code();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "PUBLIC_CCU_BUILD_FAILED error=%s\n", error.what());
        return CcuResult::CCU_E_INTERNAL;
    }
}
// Diagnostic queries do not assign/destroy the communicator-owned instance.
// The global remainder is MAX CONTIGUOUS per type, not instance-local free space.
void ResourceDiagnostic(CcuInsHandle instance, int32_t device, uint32_t die, const KernelCreator& creator,
                        const CcuKernelArg& arg) {
    const char* enabled = std::getenv("A5_CCU_RESOURCE_DIAGNOSTICS");
    if (!enabled || std::strcmp(enabled, "1") != 0) return;
    HcommCcuResDescHandle desc = 0;
    auto status = HcommCcuInsResDescCreate(die, &desc);
    if (status != CcuResult::CCU_SUCCESS) {
        std::fprintf(stderr,"PUBLIC_CCU_RESOURCE kind=create_desc die=%u status=%d\n",die,int(status));
        return;
    }
    const char* names[] = {"loop","ccu_buf","variable","address","event","ccu_thread","instruction"};
    const HcommCcuResType types[] = {HCOMM_CCU_RES_TYPE_LOOP,HCOMM_CCU_RES_TYPE_CCU_BUF,
        HCOMM_CCU_RES_TYPE_VARIABLE,HCOMM_CCU_RES_TYPE_ADDRESS,HCOMM_CCU_RES_TYPE_EVENT,
        HCOMM_CCU_RES_TYPE_CCU_THREAD,HCOMM_CCU_RES_TYPE_INSTRUCTION};
    auto report = [&](const char* kind, CcuResult result) {
        std::ostringstream line;
        line << "PUBLIC_CCU_RESOURCE kind=" << kind << " device=" << device << " pid=" << getpid()
             << " instance=" << instance << " die=" << die << " status=" << int(result);
        if (result == CcuResult::CCU_SUCCESS) for (size_t i=0;i<7;++i) {
            uint32_t value=0;
            auto queried=HcommCcuInsResDescQueryNum(desc,types[i],&value);
            if (queried==CcuResult::CCU_SUCCESS) line << " " << names[i] << "=" << value;
            else line << " " << names[i] << "_query_status=" << int(queried);
        }
        line << "\n";
        const auto record=line.str();
        // One short write prevents eight workers interleaving fields on a pipe.
        const auto written=::write(STDERR_FILENO,record.data(),record.size());
        (void)written;
    };
    try {
        report("instance_owned",HcommCcuInsQueryResDesc(instance,desc));
        report("device_max_contiguous_remaining",HcommCcuQueryRemainResDesc(desc));
        // Query builds a separate graph. Do not reuse its DSL handles in the real
        // registration transaction, and do not retry a failed registration.
        auto probe=creator(arg);
        if (probe) {
            const void* args[]={probe.get()};
            report("kernel_required",HcommCcuKernelQueryResReq(
                reinterpret_cast<const void*>(BuildKernel),args,1,desc));
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr,"PUBLIC_CCU_RESOURCE_FAILED die=%u error=%s\n",die,error.what());
    } catch (...) {
        std::fprintf(stderr,"PUBLIC_CCU_RESOURCE_FAILED die=%u error=unknown\n",die);
    }
    status=HcommCcuInsResDescDestroy(desc);
    if (status!=CcuResult::CCU_SUCCESS)
        std::fprintf(stderr,"PUBLIC_CCU_RESOURCE kind=destroy_desc die=%u status=%d\n",die,int(status));
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
    catch (const std::exception& error) {
        std::fprintf(stderr, "PUBLIC_CCU_CONSTRUCT_FAILED die=%u error=%s\n", die, error.what());
        return HCCL_E_INTERNAL;
    } catch (...) { return HCCL_E_INTERNAL; }
    if (!entry->kernel) return HCCL_E_PTR;
    CreateTrace diagnosticTrace("scope=public_register device=" + std::to_string(device) +
        " die=" + std::to_string(die) + " instance=" + std::to_string(instance));
    const char* diagnostic=std::getenv("A5_CCU_RESOURCE_DIAGNOSTICS");
    if (diagnostic && std::strcmp(diagnostic,"1")==0)
        diagnosticTrace.Call("resource_diagnostics", [&] {
            ResourceDiagnostic(instance,device,die,*creator,*arg); return HCCL_SUCCESS;
        });
    const void* args[] = {entry->kernel.get()};
    CcuKernelHandle registered = 0;
    const std::string name = "a5_public_" + std::to_string(++sequence);
    CreateTrace trace("scope=public_register device=" + std::to_string(device) +
        " die=" + std::to_string(die) + " instance=" + std::to_string(instance));
    auto ccuStatus = trace.Call("register_start", [&] { return HcommCcuKernelRegisterStart(instance); });
    if (ccuStatus != CcuResult::CCU_SUCCESS) return Status(ccuStatus);
    // Once a session starts, any failure poisons this communicator for this
    // library. Never retry a partially built instruction/resource session.
    ccuStatus = trace.Call("register_kernel", [&] { return HcommCcuKernelRegister(instance, die, name.c_str(),
        reinterpret_cast<const void*>(BuildKernel), args, 1, &registered); });
    if (ccuStatus == CcuResult::CCU_SUCCESS)
        ccuStatus = trace.Call("register_end", [&] { return HcommCcuKernelRegisterEnd(instance); });
    uint32_t actualArgs = 0;
    if (ccuStatus == CcuResult::CCU_SUCCESS)
        ccuStatus = trace.Call("get_task_args", [&] { return HcommCcuGetTaskArgsNum(registered, &actualArgs); });
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
