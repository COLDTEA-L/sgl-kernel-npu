#include "a5_uvs_source_route_provider.h"

#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "usage: provider_smoke PROVIDER BACKEND MANIFEST" << std::endl;
        return 2;
    }
    setenv("A5_UVS_EXPLICIT_ROUTE_BACKEND", argv[2], 1);
    void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        std::cerr << dlerror() << std::endl;
        return 1;
    }
    auto install = reinterpret_cast<A5UvsInstallSourceRoutesFn>(
        dlsym(handle, "A5UvsInstallSourceRoutes"));
    auto release = reinterpret_cast<A5UvsReleaseSourceRoutesFn>(
        dlsym(handle, "A5UvsReleaseSourceRoutes"));
    if (install == nullptr || release == nullptr) {
        return 1;
    }
    A5UvsSourceRoute routes[A5_UVS_MAX_SOURCE_ROUTES]{};
    uint32_t routeCount = A5_UVS_MAX_SOURCE_ROUTES;
    HcclComm fakeComm = reinterpret_cast<HcclComm>(static_cast<uintptr_t>(1));
    const int status = install(A5_UVS_SOURCE_ROUTE_ABI_VERSION, fakeComm, 0, 1,
                               argv[3], routes, &routeCount);
    if (status != 0 || routeCount != 1 || routes[0].relayPhyId != 2 ||
        routes[0].channel.remoteRank != 1) {
        std::cerr << "provider smoke failed: status=" << status
                  << " count=" << routeCount << std::endl;
        return 1;
    }
    release(fakeComm, routes, routeCount);
    std::cout << "PASS: explicit route provider manifest/install/query/release smoke" << std::endl;
    return 0;
}
