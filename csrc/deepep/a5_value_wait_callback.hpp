#pragma once

#include <acl/acl_rt.h>
#include <torch/extension.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace deep_ep {

// A deliberately small host scheduler used by the A5 puncture experiment.
// The callback itself only publishes an atomic state transition.  Runtime
// calls are made by scheduler_thread_, because CANN forbids Runtime API calls
// from aclrtLaunchHostFunc callbacks.
class A5ValueWaitCallbackScheduler
    : public std::enable_shared_from_this<A5ValueWaitCallbackScheduler> {
public:
    enum class State : int64_t {
        kIdle = 0,
        kArmed = 1,
        kArrived = 2,
        kReleased = 3,
        kDone = 4,
        kFailed = 5,
        kStopping = 6,
    };

    explicit A5ValueWaitCallbackScheduler(const torch::Tensor &path_policy);
    ~A5ValueWaitCallbackScheduler();

    A5ValueWaitCallbackScheduler(const A5ValueWaitCallbackScheduler &) = delete;
    A5ValueWaitCallbackScheduler &operator=(const A5ValueWaitCallbackScheduler &) = delete;

    // Host control-plane operations, intentionally outside graph capture.
    void arm(int64_t release_word);
    void enqueue_arrived_callback();
    void enqueue_done_callback();
    void wait_done(int64_t timeout_ms);

    // Stream tasks that are captured into the operator graph.
    void enqueue_value_wait(int64_t ready_mask);
    void enqueue_value_write(int64_t value);

    int64_t state() const;
    int64_t arrival_count() const;
    int64_t release_count() const;
    int64_t done_count() const;
    int64_t last_release_word() const;

private:
    enum class CallbackKind : uint32_t { kArrived, kDone };
    struct CallbackArg {
        std::shared_ptr<A5ValueWaitCallbackScheduler> owner;
        CallbackKind kind;
    };

    static void HostCallback(void *opaque);
    void OnCallback(CallbackKind kind) noexcept;
    void SchedulerLoop() noexcept;
    void Fail(aclError error) noexcept;
    void RecordPolicyOnCurrentStream();
    aclrtStream CurrentStream() const;

    torch::Tensor path_policy_;
    aclrtContext context_{nullptr};
    aclrtStream control_stream_{nullptr};
    std::thread scheduler_thread_;

    std::atomic<State> state_{State::kIdle};
    std::atomic<uint64_t> release_word_{0};
    std::atomic<uint64_t> last_release_word_{0};
    std::atomic<int64_t> arrival_count_{0};
    std::atomic<int64_t> release_count_{0};
    std::atomic<int64_t> done_count_{0};
    std::atomic<int32_t> runtime_error_{ACL_SUCCESS};
    mutable std::mutex mutex_;
    std::condition_variable condition_;
};

} // namespace deep_ep
