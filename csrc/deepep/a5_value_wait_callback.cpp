#include "a5_value_wait_callback.hpp"

#include <c10/core/DeviceGuard.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>

#include <chrono>
#include <memory>
#include <utility>

#include "pytorch_npu_helper.hpp"

extern "C" __attribute__((visibility("default"))) int
A5DeepEpValueWaitCallbackAbiVersion()
{
    return 1;
}

namespace deep_ep {
namespace {
constexpr uint32_t kValueWriteDefaultFlag = 0;
constexpr uint32_t kValueWaitAndFlag = 0x00000002U;

const char *StateName(A5ValueWaitCallbackScheduler::State state)
{
    switch (state) {
        case A5ValueWaitCallbackScheduler::State::kIdle: return "idle";
        case A5ValueWaitCallbackScheduler::State::kArmed: return "armed";
        case A5ValueWaitCallbackScheduler::State::kArrived: return "arrived";
        case A5ValueWaitCallbackScheduler::State::kReleased: return "released";
        case A5ValueWaitCallbackScheduler::State::kDone: return "done";
        case A5ValueWaitCallbackScheduler::State::kFailed: return "failed";
        case A5ValueWaitCallbackScheduler::State::kStopping: return "stopping";
    }
    return "unknown";
}
} // namespace

A5ValueWaitCallbackScheduler::A5ValueWaitCallbackScheduler(
    const torch::Tensor &path_policy)
    : path_policy_(path_policy)
{
    TORCH_CHECK(path_policy_.defined(), "path_policy must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(path_policy_),
                "path_policy must be an NPU tensor");
    TORCH_CHECK(path_policy_.is_contiguous(),
                "path_policy must be contiguous");
    TORCH_CHECK(path_policy_.scalar_type() == at::kLong &&
                    path_policy_.numel() == 1,
                "path_policy must contain exactly one int64 value");
    TORCH_CHECK((reinterpret_cast<uintptr_t>(path_policy_.data_ptr()) & 7U) == 0,
                "path_policy address must be 8-byte aligned");

    c10::DeviceGuard device_guard(path_policy_.device());
    aclError result = aclrtGetCurrentContext(&context_);
    TORCH_CHECK(result == ACL_SUCCESS && context_ != nullptr,
                "aclrtGetCurrentContext failed: ", result);
    result = aclrtCreateStream(&control_stream_);
    TORCH_CHECK(result == ACL_SUCCESS && control_stream_ != nullptr,
                "aclrtCreateStream for host scheduler failed: ", result);
    scheduler_thread_ = std::thread(&A5ValueWaitCallbackScheduler::SchedulerLoop,
                                    this);
}

A5ValueWaitCallbackScheduler::~A5ValueWaitCallbackScheduler()
{
    state_.store(State::kStopping, std::memory_order_release);
    condition_.notify_all();
    if (scheduler_thread_.joinable()) {
        scheduler_thread_.join();
    }
    if (control_stream_ != nullptr) {
        (void)aclrtSynchronizeStream(control_stream_);
        (void)aclrtDestroyStream(control_stream_);
        control_stream_ = nullptr;
    }
}

aclrtStream A5ValueWaitCallbackScheduler::CurrentStream() const
{
    return c10_npu::getCurrentNPUStream().stream();
}

void A5ValueWaitCallbackScheduler::RecordPolicyOnCurrentStream()
{
    const auto stream = c10_npu::getCurrentNPUStream();
    c10_npu::NPUCachingAllocator::recordStream(
        path_policy_.storage().data_ptr(), stream);
}

void A5ValueWaitCallbackScheduler::arm(int64_t release_word)
{
    const State old_state = state_.load(std::memory_order_acquire);
    TORCH_CHECK(old_state == State::kIdle || old_state == State::kDone,
                "scheduler cannot arm from state ", StateName(old_state));
    TORCH_CHECK(release_word > 0,
                "release word must be a positive signed-int64 value");
    runtime_error_.store(ACL_SUCCESS, std::memory_order_relaxed);
    release_word_.store(static_cast<uint64_t>(release_word),
                        std::memory_order_relaxed);
    state_.store(State::kArmed, std::memory_order_release);
}

void A5ValueWaitCallbackScheduler::enqueue_arrived_callback()
{
    TORCH_CHECK(state_.load(std::memory_order_acquire) == State::kArmed,
                "arrival callback requires an armed scheduler");
    auto *arg = new CallbackArg{shared_from_this(), CallbackKind::kArrived};
    const aclError result = aclrtLaunchHostFunc(CurrentStream(), HostCallback, arg);
    if (result != ACL_SUCCESS) {
        delete arg;
    }
    TORCH_CHECK(result == ACL_SUCCESS,
                "aclrtLaunchHostFunc(arrived) failed: ", result);
}

void A5ValueWaitCallbackScheduler::enqueue_done_callback()
{
    auto *arg = new CallbackArg{shared_from_this(), CallbackKind::kDone};
    const aclError result = aclrtLaunchHostFunc(CurrentStream(), HostCallback, arg);
    if (result != ACL_SUCCESS) {
        delete arg;
    }
    TORCH_CHECK(result == ACL_SUCCESS,
                "aclrtLaunchHostFunc(done) failed: ", result);
}

void A5ValueWaitCallbackScheduler::enqueue_value_wait(int64_t ready_mask)
{
    TORCH_CHECK(ready_mask > 0,
                "ready mask must be a positive signed-int64 value");
    RecordPolicyOnCurrentStream();
    const aclError result = aclrtValueWait(
        path_policy_.data_ptr(), static_cast<uint64_t>(ready_mask),
        kValueWaitAndFlag, CurrentStream());
    TORCH_CHECK(result == ACL_SUCCESS, "aclrtValueWait failed: ", result);
}

void A5ValueWaitCallbackScheduler::enqueue_value_write(int64_t value)
{
    TORCH_CHECK(value >= 0,
                "graph-side value write must fit in signed int64");
    RecordPolicyOnCurrentStream();
    const aclError result = aclrtValueWrite(
        path_policy_.data_ptr(), static_cast<uint64_t>(value),
        kValueWriteDefaultFlag, CurrentStream());
    TORCH_CHECK(result == ACL_SUCCESS, "aclrtValueWrite failed: ", result);
}

void A5ValueWaitCallbackScheduler::HostCallback(void *opaque)
{
    std::unique_ptr<CallbackArg> arg(static_cast<CallbackArg *>(opaque));
    arg->owner->OnCallback(arg->kind);
}

void A5ValueWaitCallbackScheduler::OnCallback(CallbackKind kind) noexcept
{
    if (kind == CallbackKind::kArrived) {
        State expected = State::kArmed;
        if (!state_.compare_exchange_strong(expected, State::kArrived,
                                            std::memory_order_acq_rel)) {
            runtime_error_.store(-1, std::memory_order_relaxed);
            state_.store(State::kFailed, std::memory_order_release);
        } else {
            arrival_count_.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        const State current = state_.load(std::memory_order_acquire);
        if (current != State::kArrived && current != State::kReleased) {
            runtime_error_.store(-2, std::memory_order_relaxed);
            state_.store(State::kFailed, std::memory_order_release);
        } else {
            done_count_.fetch_add(1, std::memory_order_relaxed);
            state_.store(State::kDone, std::memory_order_release);
        }
    }
    condition_.notify_all();
}

void A5ValueWaitCallbackScheduler::Fail(aclError error) noexcept
{
    runtime_error_.store(error, std::memory_order_relaxed);
    state_.store(State::kFailed, std::memory_order_release);
    condition_.notify_all();
}

void A5ValueWaitCallbackScheduler::SchedulerLoop() noexcept
{
    aclError result = aclrtSetCurrentContext(context_);
    if (result != ACL_SUCCESS) {
        Fail(result);
        return;
    }
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] {
                const State current = state_.load(std::memory_order_acquire);
                return current == State::kArrived ||
                       current == State::kStopping;
            });
        }
        if (state_.load(std::memory_order_acquire) == State::kStopping) {
            return;
        }
        const uint64_t word = release_word_.load(std::memory_order_relaxed);
        result = aclrtValueWrite(path_policy_.data_ptr(), word,
                                 kValueWriteDefaultFlag, control_stream_);
        if (result != ACL_SUCCESS) {
            Fail(result);
            continue;
        }
        last_release_word_.store(word, std::memory_order_relaxed);
        release_count_.fetch_add(1, std::memory_order_relaxed);
        State expected = State::kArrived;
        (void)state_.compare_exchange_strong(expected, State::kReleased,
                                             std::memory_order_acq_rel);
        condition_.notify_all();
    }
}

void A5ValueWaitCallbackScheduler::wait_done(int64_t timeout_ms)
{
    TORCH_CHECK(timeout_ms > 0, "timeout_ms must be positive");
    std::unique_lock<std::mutex> lock(mutex_);
    const bool ready = condition_.wait_for(
        lock, std::chrono::milliseconds(timeout_ms), [this] {
            const State current = state_.load(std::memory_order_acquire);
            // A very short graph can run the done callback between the
            // asynchronous ValueWrite returning and the scheduler publishing
            // release_count_. Wait for both observations before returning.
            return current == State::kFailed ||
                   (current == State::kDone && release_count_.load(
                       std::memory_order_acquire) >= done_count_.load(
                           std::memory_order_acquire));
        });
    TORCH_CHECK(ready, "timed out waiting for completion callback after ",
                timeout_ms, " ms; state=",
                StateName(state_.load(std::memory_order_acquire)));
    const State current = state_.load(std::memory_order_acquire);
    TORCH_CHECK(current == State::kDone,
                "scheduler failed; callback/runtime error=",
                runtime_error_.load(std::memory_order_relaxed));
}

int64_t A5ValueWaitCallbackScheduler::state() const
{
    return static_cast<int64_t>(state_.load(std::memory_order_acquire));
}

int64_t A5ValueWaitCallbackScheduler::arrival_count() const
{
    return arrival_count_.load(std::memory_order_relaxed);
}

int64_t A5ValueWaitCallbackScheduler::release_count() const
{
    return release_count_.load(std::memory_order_relaxed);
}

int64_t A5ValueWaitCallbackScheduler::done_count() const
{
    return done_count_.load(std::memory_order_relaxed);
}

int64_t A5ValueWaitCallbackScheduler::last_release_word() const
{
    return static_cast<int64_t>(
        last_release_word_.load(std::memory_order_relaxed));
}

} // namespace deep_ep
