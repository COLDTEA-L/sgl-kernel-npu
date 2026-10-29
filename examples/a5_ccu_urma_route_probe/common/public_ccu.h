#ifndef A5_PUBLIC_CCU_H
#define A5_PUBLIC_CCU_H
// Project-owned host abstractions. No hcomm::CcuKernel/CcuRep ABI is imported.
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_channel.h>
#include <hccl/hccl_res.h>
#include <hcomm/ccu/ccu_launch.h>
#include <hcomm/ccu/ccu_primitives.hpp>
#include <hcomm/ccu/ccu_res.h>
#include <memory>
#include <sstream>
#include <vector>

namespace a5ccu {
inline HcclResult Status(CcuResult status) { return static_cast<HcclResult>(status); }
namespace CcuRep {
using Variable = AscendC::ccu::Variable;
using LocalAddr = AscendC::ccu::LocalAddr;
using RemoteAddr = AscendC::ccu::RemoteAddr;
using CompletedEvent = AscendC::ccu::Event;
inline uint64_t GetTokenInfo(uint64_t address, uint64_t bytes) {
    uint64_t token = 0;
    CCU_THROW_IF_FAILED(HcommCcuGetMemToken(address, bytes, &token), "HcommCcuGetMemToken");
    return token;
}
}
class CcuKernelSignature {
public:
    template<class T> void Append(const T& value) { stream_ << value; }
private:
    std::ostringstream stream_;
};
class CcuKernelArg {
public:
    virtual ~CcuKernelArg() = default;
    virtual CcuKernelSignature GetKernelSignature() const = 0;
    std::vector<ChannelHandle> channels;
};
class CcuTaskArg { public: virtual ~CcuTaskArg() = default; };
class CcuKernel {
public:
    explicit CcuKernel(const CcuKernelArg& arg) : channels_(arg.channels) {}
    virtual ~CcuKernel() = default;
    HcclResult Build() { argumentCount_ = 0; return Algorithm(); }
    uint32_t ArgumentCount() const { return argumentCount_; }
    std::vector<uint64_t> Args(const CcuTaskArg& task) { return GeneArgs(task); }
protected:
    virtual HcclResult Algorithm() = 0;
    virtual std::vector<uint64_t> GeneArgs(const CcuTaskArg&) = 0;
    CcuRep::Variable CreateVariable() { return CcuRep::Variable(); }
    HcclResult CreateVariable(ChannelHandle channel, uint32_t index, CcuRep::Variable* out) {
        if (!out) return HCCL_E_PTR;
        *out = AscendC::ccu::GetResByChannel<CcuRep::Variable>(channel, index);
        return HCCL_SUCCESS;
    }
    CcuRep::LocalAddr CreateLocalAddr() { return CcuRep::LocalAddr(); }
    CcuRep::RemoteAddr CreateRemoteAddr() { return CcuRep::RemoteAddr(); }
    CcuRep::CompletedEvent CreateCompletedEvent() { return CcuRep::CompletedEvent(); }
    void Load(CcuRep::Variable variable) {
        CCU_THROW_IF_FAILED(AscendC::ccu::LoadArg(variable, argumentCount_++), "LoadArg");
    }
    void LoadVariable(uint64_t address, CcuRep::Variable value) {
        CCU_THROW_IF_FAILED(AscendC::ccu::Load(address, value), "LoadVariable");
    }
    void StoreVariable(CcuRep::Variable value, uint64_t address) {
        CCU_THROW_IF_FAILED(AscendC::ccu::Store(address, value), "StoreVariable");
    }
    HcclResult NotifyRecord(ChannelHandle channel, uint32_t slot, uint16_t mask) {
        return Status(AscendC::ccu::NotifyRecord(channel, slot, mask));
    }
    HcclResult NotifyRecord(ChannelHandle channel, uint32_t notify, uint32_t variable,
                           CcuRep::Variable value, uint16_t mask) {
        return Status(AscendC::ccu::WriteVariableWithNotify(channel, value, variable, notify, mask));
    }
    HcclResult NotifyWait(ChannelHandle channel, uint32_t slot, uint16_t mask) {
        return Status(AscendC::ccu::NotifyWait(channel, slot, mask));
    }
    HcclResult WriteNb(ChannelHandle channel, CcuRep::RemoteAddr dst, CcuRep::LocalAddr src,
                       CcuRep::Variable bytes, CcuRep::CompletedEvent event) {
        return Status(AscendC::ccu::Write(channel, dst, src, bytes, event));
    }
    template<class Destination, class Source>
    HcclResult LocalCopyNb(Destination dst, Source src, CcuRep::Variable bytes, CcuRep::CompletedEvent event) {
        return Status(AscendC::ccu::LocalCopy(dst, src, bytes, event));
    }
    HcclResult WaitEvent(CcuRep::CompletedEvent event) {
        return Status(AscendC::ccu::EventWait(event));
    }
    std::vector<ChannelHandle> channels_;
private:
    uint32_t argumentCount_ = 0;
};
using KernelCreator = std::unique_ptr<CcuKernel> (*)(const CcuKernelArg&);
HcclResult RegisterKernel(HcclComm comm, uint32_t die, uint64_t* handle,
                          const KernelCreator* creator, const CcuKernelArg* arg);
HcclResult LaunchKernel(HcclComm comm, ThreadHandle thread, uint64_t handle, const CcuTaskArg* task);
} // namespace a5ccu
#endif
