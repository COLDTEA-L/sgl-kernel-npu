#include "source_route_provider.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace a5_ccu_urma_probe {

HcclResult LoadProviderRoutes(HcclComm comm, uint32_t rank, uint32_t peer,
                              std::vector<A5UvsSourceRoute> *routes)
{
    if (routes == nullptr) {
        return HCCL_E_PTR;
    }
    const char *manifest = std::getenv("A5_CCU_SOURCE_ROUTE_MANIFEST");
    if (manifest == nullptr || manifest[0] == '\0') {
        return HCCL_E_NOT_FOUND;
    }
    const char *providerPath = std::getenv("A5_CCU_SOURCE_ROUTE_PROVIDER");
    if (providerPath == nullptr || providerPath[0] == '\0') {
        std::fprintf(stderr,
            "[A5 CCU URMA] source-route manifest is set but no provider was supplied; "
            "set A5_CCU_SOURCE_ROUTE_PROVIDER to the platform UVS/HIXL plugin\n");
        return HCCL_E_NOT_SUPPORT;
    }
    void *handle = dlopen(providerPath, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        std::fprintf(stderr, "[A5 CCU URMA] dlopen source-route provider failed: %s\n", dlerror());
        return HCCL_E_NOT_FOUND;
    }
    auto install = reinterpret_cast<A5UvsInstallSourceRoutesFn>(
        dlsym(handle, "A5UvsInstallSourceRoutes"));
    if (install == nullptr) {
        std::fprintf(stderr,
            "[A5 CCU URMA] provider lacks A5UvsInstallSourceRoutes ABI v%u\n",
            A5_UVS_SOURCE_ROUTE_ABI_VERSION);
        dlclose(handle);
        return HCCL_E_NOT_SUPPORT;
    }
    routes->resize(A5_UVS_MAX_SOURCE_ROUTES);
    uint32_t routeCount = A5_UVS_MAX_SOURCE_ROUTES;
    const int providerStatus = install(A5_UVS_SOURCE_ROUTE_ABI_VERSION, comm, rank, peer,
                                       manifest, routes->data(), &routeCount);
    if (providerStatus != 0 || routeCount == 0 || routeCount > A5_UVS_MAX_SOURCE_ROUTES) {
        std::fprintf(stderr,
            "[A5 CCU URMA] provider failed to install forwarding routes: status=%d count=%u\n",
            providerStatus, routeCount);
        routes->clear();
        dlclose(handle);
        return HCCL_E_RUNTIME;
    }
    routes->resize(routeCount);
    /* Keep the provider loaded: channel descriptors and forwarding rules stay
       valid for the lifetime of the process/communicator. */
    std::printf("[A5 CCU URMA] provider installed %u explicit source route(s) from %s\n",
                routeCount, manifest);
    std::fflush(stdout);
    return HCCL_SUCCESS;
}

} // namespace a5_ccu_urma_probe
