#ifndef A5_CCU_EP8_DFX_H
#define A5_CCU_EP8_DFX_H
#include <hccl/hccl_diag.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <memory>

namespace a5_ccu_urma_probe {
// Use the installed SDK's public struct, never a guessed private DfxOpInfo ABI.
// This follows HCCL's native op registration before submitting CCU work. The
// registration API copies metadata into a shared DfxOpInfo owned by HCCL tasks.
class Ep8Dfx {
public:
    static HcclResult Create(HcclComm comm, uint64_t handle, std::shared_ptr<Ep8Dfx> *out)
    {
        if (!comm || !out) return HCCL_E_PTR;
        auto dfx = std::shared_ptr<Ep8Dfx>(new Ep8Dfx);
        auto name = reinterpret_cast<GetName>(Symbol("HcclGetCommName"));
        dfx->reg_ = reinterpret_cast<Register>(Symbol("HcclDfxRegOpInfoByCommId"));
        if (!name || !dfx->reg_) return HCCL_E_NOT_SUPPORT;
        auto status = name(comm, dfx->name_);
        if (status != HCCL_SUCCESS) return status;
        if (!dfx->name_[0] || !std::memchr(dfx->name_, 0, sizeof(dfx->name_))) return HCCL_E_PARA;
        dfx->comm_ = comm; dfx->handle_ = handle;
        *out = std::move(dfx);
        return HCCL_SUCCESS;
    }
    // Invoked under the peer-plan submit mutex. Re-register on EVERY call:
    // native collectives may have replaced this communicator's current op info.
    template<class Submit>
    HcclResult Run(uint32_t rank, uint64_t send, uint64_t recv,
                   uint64_t sendCapacity, uint64_t recvCapacity, uint64_t bytes, Submit submit)
    {
        HcclDfxOpInfo info{};
        info.opMode = 1; // HCCL OpMode::OPBASE (OFFLOAD=0), as in native eager ops.
        info.opType = HCCL_CMD_ALLTOALLV;
        info.reduceOp = HCCL_REDUCE_RESERVED;
        info.dataType = HCCL_DATA_TYPE_UINT8; // raw row payload; dataCount is bytes
        info.dataCount = bytes;
        info.engine = COMM_ENGINE_CCU;
        std::snprintf(info.algTag, sizeof(info.algTag), "a5_ep8_alltoallv_%llu",
                      static_cast<unsigned long long>(handle_));
        SetMemory(info, send, recv, sendCapacity, recvCapacity, 0);
        const uint64_t sequence = ++attempts_;
        auto status = reg_(name_, &info);
        if (status != HCCL_SUCCESS) {
            Log(rank, sequence, "register_op", status);
            return status; // no stream/event/CCU submission after registration failure
        }
        if (sequence == 1) Log(rank, sequence, "registered", status);
        status = submit(); // exactly once, including when launch reports a DFX failure
        if (status != HCCL_SUCCESS) Log(rank, sequence, "submit", status);
        return status;
    }
private:
    using GetName = HcclResult (*)(HcclComm, char *);
    using Register = HcclResult (*)(char *, void *);
    static void *Symbol(const char *name)
    {
        void *symbol = dlsym(RTLD_DEFAULT, name);
        if (symbol) return symbol;
        static void *library = dlopen("libhcomm.so", RTLD_NOW | RTLD_LOCAL);
        return library ? dlsym(library, name) : nullptr;
    }
    // Later 9.1 headers expose memory fields in formerly reserved space. Use
    // named fields when present; leave reserved bytes zero on earlier headers.
    template<class Info>
    static auto SetMemory(Info &i, uint64_t s, uint64_t r, uint64_t sc, uint64_t rc, int)
        -> decltype(i.inputMemAddr = s, i.outputMemAddr = r,
                    i.inputMemSize = sc, i.outputMemSize = rc, void())
    { i.inputMemAddr = s; i.outputMemAddr = r; i.inputMemSize = sc; i.outputMemSize = rc; }
    template<class Info>
    static void SetMemory(Info &, uint64_t, uint64_t, uint64_t, uint64_t, long) {}
    void Log(uint32_t rank, uint64_t sequence, const char *stage, HcclResult status) const
    {
        std::fprintf(stderr, "EP8_DFX rank=%u handle=%llu sequence=%llu comm=%p group=%s stage=%s status=%d\n",
                     rank, static_cast<unsigned long long>(handle_),
                     static_cast<unsigned long long>(sequence), comm_, name_, stage, static_cast<int>(status));
        std::fflush(stderr);
    }
    char name_[COMM_NAME_MAX_LENGTH]{};
    HcclComm comm_ = nullptr;
    Register reg_ = nullptr;
    uint64_t handle_ = 0, attempts_ = 0;
};
} // namespace a5_ccu_urma_probe
#endif
