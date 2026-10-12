#ifndef A5_CCU_SELF_COPY_LANE_H
#define A5_CCU_SELF_COPY_LANE_H
#include <acl/acl_rt.h>
#include <hccl/hccl_types.h>
#include <memory>

namespace a5_ccu_urma_probe {
// Owned by a (plan, execution stream), not constructed per invocation.
// The caller stream joins the copy branch, so tensor allocator lifetimes
// recorded on that caller stream also cover copy-stream accesses.
class SelfCopyLane {
public:
    static HcclResult Create(aclrtStream main, std::shared_ptr<SelfCopyLane> *out);
    ~SelfCopyLane();
    HcclResult Begin(void *destination, const void *source, uint64_t bytes);
    HcclResult Join(HcclResult launchStatus);
    HcclResult Close(); // outside capture, after all graph users are released
    bool overlap() const { return overlap_; }
    SelfCopyLane(const SelfCopyLane &) = delete;
    SelfCopyLane &operator=(const SelfCopyLane &) = delete;
private:
    SelfCopyLane() = default;
    aclrtStream main_ = nullptr, copy_ = nullptr;
    aclrtEvent ready_ = nullptr, done_ = nullptr;
    int32_t device_ = -1;
    bool overlap_ = false, pending_ = false, failed_ = false, closed_ = false;
    void *destination_ = nullptr;
    const void *source_ = nullptr;
    uint64_t bytes_ = 0;
    HcclResult Fail(const char *phase, aclError status, bool copySubmitted);
};

class SelfCopyGuard {
public:
    explicit SelfCopyGuard(const std::shared_ptr<SelfCopyLane> &lane) : lane_(lane) {}
    SelfCopyGuard(const SelfCopyGuard &) = delete;
    SelfCopyGuard &operator=(const SelfCopyGuard &) = delete;
    ~SelfCopyGuard() { if (started_) (void)Finish(HCCL_E_RUNTIME); }
    HcclResult Begin(void *destination, const void *source, uint64_t bytes)
    {
        if (!lane_) return HCCL_E_INTERNAL;
        const auto status = lane_->Begin(destination, source, bytes);
        started_ = status == HCCL_SUCCESS;
        return status;
    }
    HcclResult Finish(HcclResult status)
    {
        if (!started_) return status;
        started_ = false;
        return lane_->Join(status);
    }
private:
    std::shared_ptr<SelfCopyLane> lane_;
    bool started_ = false;
};
} // namespace a5_ccu_urma_probe
#endif
