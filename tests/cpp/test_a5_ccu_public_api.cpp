// Host control-plane test with the real installed public headers and mocked
// vendor entry points. It does not initialize ACL or model NPU execution.
#include "../../examples/a5_ccu_urma_route_probe/common/public_ccu.h"
#include <cassert>
#include <cstdio>
#include <vector>
namespace {
int device = 0, queries = 0, starts = 0, registrations = 0, ends = 0, launches = 0;
uint32_t registeredDie = 0;
uint64_t nextHandle = 20, lastValue = 0;
CcuResult registrationStatus = CcuResult::CCU_SUCCESS;
CcuResult launchStatus = CcuResult::CCU_SUCCESS;
bool badArgCount = false;
std::vector<int> order;
struct Arg : a5ccu::CcuKernelArg {
    a5ccu::CcuKernelSignature GetKernelSignature() const override { return {}; }
};
struct Task : a5ccu::CcuTaskArg { uint64_t value = 0; };
class Kernel : public a5ccu::CcuKernel {
public:
    using a5ccu::CcuKernel::CcuKernel;
protected:
    HcclResult Algorithm() override { auto v = CreateVariable(); Load(v); return HCCL_SUCCESS; }
    std::vector<uint64_t> GeneArgs(const a5ccu::CcuTaskArg& arg) override {
        const auto* task = dynamic_cast<const Task*>(&arg);
        return task ? std::vector<uint64_t>{task->value} : std::vector<uint64_t>{};
    }
};
std::unique_ptr<a5ccu::CcuKernel> Creator(const a5ccu::CcuKernelArg& arg) {
    return std::unique_ptr<a5ccu::CcuKernel>(new Kernel(arg));
}
}
extern "C" aclError aclrtGetDevice(int32_t* out) { *out = device; return ACL_SUCCESS; }
extern "C" HcclResult HcclCommQueryCcuIns(HcclComm, CcuInsHandle* out, uint32_t* count) {
    ++queries; *out = 77; *count = 1; return HCCL_SUCCESS;
}
extern "C" CcuResult CcuVariableAlloc(CcuVariableHandle* out) { *out = 100; return CcuResult::CCU_SUCCESS; }
extern "C" CcuResult CcuLoadArg(CcuVariableHandle, uint32_t index) { assert(index == 0); return CcuResult::CCU_SUCCESS; }
extern "C" CcuResult HcommCcuKernelRegisterStart(CcuInsHandle instance) {
    assert(instance == 77); ++starts; order.push_back(1); return CcuResult::CCU_SUCCESS;
}
extern "C" CcuResult HcommCcuKernelRegister(CcuInsHandle instance, uint32_t die, const char*,
    const void* function, const void** args, uint32_t count, CcuKernelHandle* handle) {
    assert(instance == 77 && count == 1 && args && args[0]);
    ++registrations; registeredDie = die; order.push_back(2);
    if (registrationStatus != CcuResult::CCU_SUCCESS) return registrationStatus;
    auto callback = reinterpret_cast<CcuResult (*)(::CcuKernelArg)>(const_cast<void*>(function));
    auto status = callback(const_cast<void*>(args[0]));
    *handle = nextHandle++;
    return status;
}
extern "C" CcuResult HcommCcuKernelRegisterEnd(CcuInsHandle instance) {
    assert(instance == 77); ++ends; order.push_back(3); return CcuResult::CCU_SUCCESS;
}
extern "C" CcuResult HcommCcuGetTaskArgsNum(CcuKernelHandle, uint32_t* out) {
    *out = badArgCount ? 2 : 1; return CcuResult::CCU_SUCCESS;
}
extern "C" CcuResult HcommCcuKernelLaunch(ThreadHandle thread, CcuKernelHandle, const void* args, uint32_t count) {
    assert(thread == 8 && count == 1); ++launches;
    lastValue = static_cast<const uint64_t*>(args)[0]; return launchStatus;
}
int main() {
    auto comm = reinterpret_cast<HcclComm>(0x10), other = reinterpret_cast<HcclComm>(0x20);
    Arg arg; a5ccu::KernelCreator creator = Creator; uint64_t handle = 0;
    assert(a5ccu::RegisterKernel(comm, 1, &handle, &creator, &arg) == HCCL_SUCCESS);
    assert(registeredDie == 1 && queries == 1 && starts == 1 && registrations == 1 && ends == 1);
    assert((order == std::vector<int>{1,2,3}));
    for (uint64_t i = 0; i < 100; ++i) {
        Task task; task.value = i;
        assert(a5ccu::LaunchKernel(comm, 8, handle, &task) == HCCL_SUCCESS);
        assert(lastValue == i);
    }
    assert(launches == 100 && queries == 1 && starts == 1 && registrations == 1 && ends == 1);
    Task task;
    assert(a5ccu::LaunchKernel(other, 8, handle, &task) == HCCL_E_PARA);
    device = 1;
    assert(a5ccu::LaunchKernel(comm, 8, handle, &task) == HCCL_E_NOT_FOUND);
    device = 0;
    uint64_t second = 0;
    assert(a5ccu::RegisterKernel(comm, 0, &second, &creator, &arg) == HCCL_SUCCESS && second != handle);
    assert(a5ccu::LaunchKernel(comm, 8, handle, &task) == HCCL_SUCCESS); // old binding remains valid
    launchStatus = CcuResult::CCU_E_PTR;
    const auto oldLaunches = launches;
    assert(a5ccu::LaunchKernel(comm, 8, handle, &task) == HCCL_E_PTR);
    assert(launches == oldLaunches + 1); // no retry after a failed submission
    launchStatus = CcuResult::CCU_SUCCESS;
    assert(a5ccu::RegisterKernel(comm, 2, &second, &creator, &arg) == HCCL_E_PARA);
    badArgCount = true;
    assert(a5ccu::RegisterKernel(other, 0, &second, &creator, &arg) == HCCL_E_PARA);
    auto oldStarts = starts;
    assert(a5ccu::RegisterKernel(other, 0, &second, &creator, &arg) == HCCL_E_UNAVAIL);
    assert(starts == oldStarts); // no retry after partial registration
    badArgCount = false;
    registrationStatus = CcuResult::CCU_E_PTR;
    auto failed = reinterpret_cast<HcclComm>(0x30);
    const auto oldEnds = ends;
    assert(a5ccu::RegisterKernel(failed, 0, &second, &creator, &arg) == HCCL_E_PTR);
    assert(ends == oldEnds && second == 0);
    std::puts("PUBLIC_CCU_HOST_TEST_PASS launches=102 single_registration_reuse=100 failures_preserved=1");
}
