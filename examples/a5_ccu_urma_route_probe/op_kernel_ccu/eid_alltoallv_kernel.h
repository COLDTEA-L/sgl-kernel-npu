#pragma once
#include "grouped_local_copy_kernel.h"
#include "../common/eid_alltoallv_layout.h"
#include "../common/eid_register_stage.h"
namespace a5_ccu_urma_probe {
class EidAllToAllVKernelArg : public a5ccu::CcuKernelArg {
public:
    uint32_t rank;
    EidRegisterStage stage;
    bool ready;
    std::vector<EidChannel> paths;
    EidAllToAllVKernelArg(const std::vector<ChannelHandle>& input, uint32_t r, const std::vector<EidChannel>& p,
                          EidRegisterStage buildStage = EidRegisterStage::FULL, bool readySchedule = false)
        : rank(r), stage(buildStage), ready(readySchedule), paths(p) { channels = input; }
    a5ccu::CcuKernelSignature GetKernelSignature() const override;
};
class EidAllToAllVTaskArg : public a5ccu::CcuTaskArg {
public:
    uint64_t input, output, inputToken, outputToken;
    EidAllToAllVLayout layout;
    EidAllToAllVTaskArg(uint64_t in, uint64_t out, uint64_t it, uint64_t ot, const EidAllToAllVLayout& l)
        : input(in), output(out), inputToken(it), outputToken(ot), layout(l) {}
};
class EidAllToAllVKernel : public GroupedLocalCopyKernel {
public:
    explicit EidAllToAllVKernel(const a5ccu::CcuKernelArg& arg)
        : GroupedLocalCopyKernel(arg), rank_(dynamic_cast<const EidAllToAllVKernelArg&>(arg).rank),
          paths_(dynamic_cast<const EidAllToAllVKernelArg&>(arg).paths),
          stage_(dynamic_cast<const EidAllToAllVKernelArg&>(arg).stage),
          ready_(dynamic_cast<const EidAllToAllVKernelArg&>(arg).ready) {}
protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const a5ccu::CcuTaskArg&) override;
private:
    uint32_t rank_;
    std::vector<EidChannel> paths_;
    EidRegisterStage stage_;
    bool ready_;
};
std::unique_ptr<a5ccu::CcuKernel> CreateEidAllToAllVKernel(const a5ccu::CcuKernelArg&);
}
