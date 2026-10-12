#include "self_copy_lane.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
using CreateEventEx = aclError (*)(aclrtEvent *, uint32_t);
CreateEventEx EventCreator()
{
    static auto create = []() -> CreateEventEx {
        auto symbol = dlsym(RTLD_DEFAULT, "aclrtCreateEventExWithFlag");
        if (!symbol) {
            // Keep the provider loaded as long as event handles may exist.
            static void *library = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
            if (library) symbol = dlsym(library, "aclrtCreateEventExWithFlag");
        }
        return reinterpret_cast<CreateEventEx>(symbol);
    }();
    return create;
}
} // namespace

extern "C" __attribute__((visibility("default"))) int A5CcuCopyOverlapAbiVersion() { return 1; }
extern "C" __attribute__((visibility("default"))) int A5CcuCopyOverlapSupported()
{
    return EventCreator() != nullptr; // symbol check only, not a hardware proof
}

namespace a5_ccu_urma_probe {
HcclResult GetCcuSelfCopyMode(bool *ccu)
{
    if (!ccu) return HCCL_E_PTR;
    const char *mode = std::getenv("A5_CCU_SELF_COPY_MODE");
    *ccu = mode && std::strcmp(mode, "ccu") == 0;
    if (!mode || *ccu || std::strcmp(mode, "overlap") == 0 || std::strcmp(mode, "serial") == 0)
        return HCCL_SUCCESS;
    std::fprintf(stderr, "invalid A5_CCU_SELF_COPY_MODE; use ccu, serial or overlap\n");
    return HCCL_E_PARA;
}

extern "C" __attribute__((visibility("default"))) int A5CcuLocalRemoteAbiVersion() { return 1; }

HcclResult SelfCopyLane::Create(aclrtStream main, std::shared_ptr<SelfCopyLane> *out)
{
    if (!main || !out) return HCCL_E_PTR;
    auto lane = std::shared_ptr<SelfCopyLane>(new SelfCopyLane);
    const char *mode = std::getenv("A5_CCU_SELF_COPY_MODE");
    if (!mode || std::strcmp(mode, "overlap") == 0) lane->overlap_ = true;
    else if (std::strcmp(mode, "serial") != 0) return HCCL_E_PARA;
    lane->main_ = main;
    if (aclrtGetDevice(&lane->device_) != ACL_SUCCESS) return HCCL_E_RUNTIME;
    if (lane->overlap_) {
        const auto create = EventCreator();
        if (!create) {
            std::fprintf(stderr, "COPY_OVERLAP_UNSUPPORTED aclrtCreateEventExWithFlag is required; no unsafe fallback\n");
            return HCCL_E_NOT_SUPPORT;
        }
        if (aclrtCreateStream(&lane->copy_) != ACL_SUCCESS ||
            create(&lane->ready_, ACL_EVENT_SYNC) != ACL_SUCCESS ||
            create(&lane->done_, ACL_EVENT_SYNC) != ACL_SUCCESS) {
            // No work was submitted; safely reclaim a partially built lane.
            if (lane->ready_) (void)aclrtDestroyEvent(lane->ready_);
            if (lane->done_) (void)aclrtDestroyEvent(lane->done_);
            if (lane->copy_) (void)aclrtDestroyStream(lane->copy_);
            lane->closed_ = true;
            return HCCL_E_RUNTIME;
        }
    }
    std::printf("SELF_COPY_LANE mode=%s main=%p copy=%p reusable_events=%u\n",
        lane->overlap_ ? "overlap" : "serial", main, lane->copy_, lane->overlap_ ? 2U : 0U);
    std::fflush(stdout);
    *out = std::move(lane);
    return HCCL_SUCCESS;
}

HcclResult SelfCopyLane::Fail(const char *phase, aclError status, bool copySubmitted)
{
    failed_ = true;
    std::fprintf(stderr, "COPY_OVERLAP_ERROR phase=%s status=%d; lane poisoned\n", phase, status);
    // Error-only drain: never leave a detached copy touching freed tensors.
    // A device/runtime failure can also make the drain fail; retain the error.
    if (copySubmitted && copy_) (void)aclrtSynchronizeStream(copy_);
    pending_ = false;
    return HCCL_E_RUNTIME;
}

HcclResult SelfCopyLane::Begin(void *destination, const void *source, uint64_t bytes)
{
    if (!destination || !source || !bytes || closed_ || failed_ || pending_) return HCCL_E_PARA;
    if (!overlap_) {
        const auto status = aclrtMemcpyAsync(destination, bytes, source, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, main_);
        return status == ACL_SUCCESS ? HCCL_SUCCESS : Fail("serial_copy", status, false);
    }
    // Ex events capture a record's progress at wait submission. A later record
    // cannot change an already submitted wait: no reset/reuse race in batches.
    const auto status = aclrtRecordEvent(ready_, main_);
    if (status != ACL_SUCCESS) return Fail("record_input_ready", status, false);
    destination_ = destination; source_ = source; bytes_ = bytes;
    // Defer actual copy submission until AFTER CCU has been submitted. A tiny
    // copy should not run to completion while the CPU is preparing CCU launch.
    pending_ = true;
    return HCCL_SUCCESS;
}

HcclResult SelfCopyLane::Join(HcclResult launchStatus)
{
    if (!overlap_) return launchStatus;
    if (!pending_) return HCCL_E_INTERNAL;
    if (launchStatus != HCCL_SUCCESS) {
        pending_ = false; // no detached copy exists on this failed launch
        failed_ = true;
        std::fprintf(stderr, "COPY_OVERLAP_ERROR phase=ccu_submit status=%d; lane poisoned\n", launchStatus);
        return launchStatus;
    }
    auto status = aclrtStreamWaitEvent(copy_, ready_);
    if (status != ACL_SUCCESS) return Fail("copy_wait_input", status, false);
    status = aclrtMemcpyAsync(destination_, bytes_, source_, bytes_, ACL_MEMCPY_DEVICE_TO_DEVICE, copy_);
    if (status != ACL_SUCCESS) return Fail("async_copy", status, true);
    status = aclrtRecordEvent(done_, copy_);
    if (status != ACL_SUCCESS) return Fail("record_copy_done", status, true);
    status = aclrtStreamWaitEvent(main_, done_);
    if (status != ACL_SUCCESS) return Fail("main_join_copy", status, true);
    pending_ = false;
    return launchStatus;
}

HcclResult SelfCopyLane::Close()
{
    if (closed_) return HCCL_SUCCESS;
    if (pending_) return HCCL_E_PARA;
    int32_t device = -1;
    if (aclrtGetDevice(&device) != ACL_SUCCESS || device != device_) return HCCL_E_PARA;
    // Not on the hot path. Main drain includes the join and all consumer waits.
    // Call only outside capture, with no live graph referencing these events.
    if (main_ && aclrtSynchronizeStream(main_) != ACL_SUCCESS) return HCCL_E_RUNTIME;
    if (copy_ && aclrtSynchronizeStream(copy_) != ACL_SUCCESS) return HCCL_E_RUNTIME;
    if (ready_ && aclrtDestroyEvent(ready_) != ACL_SUCCESS) return HCCL_E_RUNTIME;
    ready_ = nullptr;
    if (done_ && aclrtDestroyEvent(done_) != ACL_SUCCESS) return HCCL_E_RUNTIME;
    done_ = nullptr;
    if (copy_ && aclrtDestroyStream(copy_) != ACL_SUCCESS) return HCCL_E_RUNTIME;
    copy_ = nullptr;
    closed_ = true;
    return HCCL_SUCCESS;
}

SelfCopyLane::~SelfCopyLane()
{
    // Normal callers explicitly Close while their execution stream is alive.
    // Do not invoke ACL from late process/static teardown: the runtime/stream
    // may already have been destroyed. Unreleased handles are reclaimed by
    // process exit; they are not allocated per invocation.
    if (!closed_ && (copy_ || ready_ || done_)) {
        std::fprintf(stderr, "SELF_COPY_LANE_NOT_CLOSED main=%p; release plan before runtime teardown\n", main_);
    }
}
} // namespace a5_ccu_urma_probe
