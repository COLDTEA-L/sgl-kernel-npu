#ifndef A5_CCU_EP8_DFX_H
#define A5_CCU_EP8_DFX_H
#include <hccl/hccl_types.h>
#include <memory>
namespace a5_ccu_urma_probe {
// Public HcommCcuKernelLaunch owns its reporting path. Do not inject the
// private 9.1 HcclDfxRegOpInfoByCommId ABI into a 9.2/9.3 process.
class Ep8Dfx {
public:
    static HcclResult Create(HcclComm comm, uint64_t, std::shared_ptr<Ep8Dfx>* out) {
        if (!comm || !out) return HCCL_E_PTR;
        *out = std::make_shared<Ep8Dfx>(); return HCCL_SUCCESS;
    }
    template<class Submit>
    HcclResult Run(uint32_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, Submit submit) {
        return submit(); // exactly once, preserving launch failures
    }
};
}
#endif
