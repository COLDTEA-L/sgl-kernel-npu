// Compile/link/load probe only: does not initialize ACL or allocate an NPU.
#include <acl/acl_rt.h>
#include <dlfcn.h>
#include <cstdio>
#include <type_traits>

using ValueOp = aclError (*)(void *, uint64_t, uint32_t, aclrtStream);
using HostOp = aclError (*)(aclrtStream, aclrtHostFunc, void *);
static_assert(std::is_same<decltype(&aclrtValueWait), ValueOp>::value, "ValueWait ABI");
static_assert(std::is_same<decltype(&aclrtValueWrite), ValueOp>::value, "ValueWrite ABI");
static_assert(std::is_same<decltype(&aclrtLaunchHostFunc), HostOp>::value, "HostFunc ABI");

int main()
{
    struct Symbol { const char *name; void *address; };
    const Symbol symbols[] = {
        {"aclrtValueWait", reinterpret_cast<void *>(&aclrtValueWait)},
        {"aclrtValueWrite", reinterpret_cast<void *>(&aclrtValueWrite)},
        {"aclrtLaunchHostFunc", reinterpret_cast<void *>(&aclrtLaunchHostFunc)},
    };
    for (const auto &symbol : symbols) {
        Dl_info info{};
        if (!symbol.address || !dladdr(symbol.address, &info) || !info.dli_fname) {
            std::fprintf(stderr, "VALUEWAIT_RUNTIME_ABI_FAIL symbol=%s\n", symbol.name);
            return 1;
        }
        std::printf("VALUEWAIT_RUNTIME_LIBRARY symbol=%s path=%s\n", symbol.name, info.dli_fname);
    }
    std::puts("VALUEWAIT_RUNTIME_ABI_PASS (headers/link/load only; no device or graph validation)");
    return 0;
}
