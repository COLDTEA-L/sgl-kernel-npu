#ifndef A5_CCU_THREAD_EVENT_BRIDGE_H
#define A5_CCU_THREAD_EVENT_BRIDGE_H
#include <acl/acl_rt.h>
#include <hccl/hccl_types.h>
#include <dlfcn.h>
#include <cstdio>
#include <memory>

namespace a5_ccu_urma_probe {
// EP8 only: bridge a caller ACL stream to the stream bound to its CCU thread.
// Owns two Ex events for the plan lifetime, not per launch. All submissions are
// serialized by the peer-plan submit mutex; caller order includes the final join.
class ThreadEventBridge {
public:
    static HcclResult Create(aclrtStream main, aclrtStream route,
                            std::shared_ptr<ThreadEventBridge> *out)
    {
        if (!main || !route || main == route || !out) return HCCL_E_PARA;
        using CreateEvent = aclError (*)(aclrtEvent *, uint32_t);
        auto create = reinterpret_cast<CreateEvent>(dlsym(RTLD_DEFAULT, "aclrtCreateEventExWithFlag"));
        if (!create) {
            static void *library = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
            if (library) create = reinterpret_cast<CreateEvent>(dlsym(library, "aclrtCreateEventExWithFlag"));
        }
        if (!create) return HCCL_E_NOT_SUPPORT;
        auto bridge = std::shared_ptr<ThreadEventBridge>(new ThreadEventBridge);
        bridge->main_ = main; bridge->route_ = route;
        if (aclrtGetDevice(&bridge->device_) != ACL_SUCCESS) return HCCL_E_RUNTIME;
        if (create(&bridge->ready_, ACL_EVENT_SYNC) != ACL_SUCCESS ||
            create(&bridge->done_, ACL_EVENT_SYNC) != ACL_SUCCESS) {
            if (bridge->ready_) (void)aclrtDestroyEvent(bridge->ready_);
            if (bridge->done_) (void)aclrtDestroyEvent(bridge->done_);
            return HCCL_E_RUNTIME;
        }
        *out = std::move(bridge);
        return HCCL_SUCCESS;
    }
    HcclResult Begin()
    {
        if (closed_ || poisoned_ || pending_) return HCCL_E_PARA;
        auto status = aclrtRecordEvent(ready_, main_);
        if (status != ACL_SUCCESS) return Fail("record_ready", status);
        status = aclrtStreamWaitEvent(route_, ready_);
        if (status != ACL_SUCCESS) return Fail("route_wait_ready", status);
        pending_ = true;
        return HCCL_SUCCESS;
    }
    HcclResult Finish(HcclResult launchStatus)
    {
        if (!pending_) return HCCL_E_PARA;
        pending_ = false;
        if (launchStatus != HCCL_SUCCESS) { poisoned_ = true; return launchStatus; }
        // Publish BEFORE submitting the consumer wait. A failed record must
        // not leave the caller queued behind a notification that cannot arrive.
        auto status = aclrtRecordEvent(done_, route_);
        if (status != ACL_SUCCESS) return Fail("record_done", status);
        status = aclrtStreamWaitEvent(main_, done_);
        if (status != ACL_SUCCESS) return Fail("main_wait_done", status);
        return HCCL_SUCCESS;
    }
    HcclResult Close()
    {
        if (closed_) return HCCL_SUCCESS;
        int32_t device = -1;
        if (pending_ || aclrtGetDevice(&device) != ACL_SUCCESS || device != device_) return HCCL_E_PARA;
        if (aclrtSynchronizeStream(main_) != ACL_SUCCESS || aclrtSynchronizeStream(route_) != ACL_SUCCESS)
            return HCCL_E_RUNTIME;
        if (ready_ && aclrtDestroyEvent(ready_) != ACL_SUCCESS) return HCCL_E_RUNTIME;
        ready_ = nullptr;
        if (done_ && aclrtDestroyEvent(done_) != ACL_SUCCESS) return HCCL_E_RUNTIME;
        done_ = nullptr; closed_ = true;
        return HCCL_SUCCESS;
    }
    // Explicit Close while streams/runtime live; never invoke ACL from static teardown.
    ThreadEventBridge(const ThreadEventBridge &) = delete;
    ThreadEventBridge &operator=(const ThreadEventBridge &) = delete;
private:
    ThreadEventBridge() = default;
    HcclResult Fail(const char *stage, aclError status)
    {
        poisoned_ = true; pending_ = false;
        std::fprintf(stderr, "EP8_EVENT_BRIDGE_FAILED stage=%s acl_status=%d main=%p route=%p\n",
                     stage, status, main_, route_);
        std::fflush(stderr);
        return HCCL_E_RUNTIME;
    }
    aclrtStream main_ = nullptr, route_ = nullptr;
    aclrtEvent ready_ = nullptr, done_ = nullptr;
    int32_t device_ = -1;
    bool closed_ = false, poisoned_ = false, pending_ = false;
};
} // namespace a5_ccu_urma_probe
#endif
