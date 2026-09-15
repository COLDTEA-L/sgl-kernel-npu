#include "source_route_provider.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace a5_ccu_urma_probe {
namespace {
struct ProviderInstallation {
    void *handle = nullptr;
    A5UvsReleaseSourceRoutesFn release = nullptr;
    HcclComm comm = nullptr;
    std::vector<A5UvsSourceRoute> routes;
};

std::mutex g_providerMutex;
std::vector<ProviderInstallation> g_providerInstallations;
bool g_cleanupRegistered = false;

void ReleaseProviderRoutes()
{
    std::lock_guard<std::mutex> lock(g_providerMutex);
    for (auto it = g_providerInstallations.rbegin();
         it != g_providerInstallations.rend(); ++it) {
        if (it->release != nullptr) {
            it->release(it->comm, it->routes.data(),
                        static_cast<uint32_t>(it->routes.size()));
        }
        if (it->handle != nullptr) {
            dlclose(it->handle);
        }
    }
    g_providerInstallations.clear();
}
} // namespace

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
    auto release = reinterpret_cast<A5UvsReleaseSourceRoutesFn>(
        dlsym(handle, "A5UvsReleaseSourceRoutes"));
    if (release == nullptr) {
        std::fprintf(stderr,
            "[A5 CCU URMA] provider lacks A5UvsReleaseSourceRoutes ABI v%u\n",
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
    {
        std::lock_guard<std::mutex> lock(g_providerMutex);
        if (!g_cleanupRegistered) {
            std::atexit(ReleaseProviderRoutes);
            g_cleanupRegistered = true;
        }
        g_providerInstallations.push_back({handle, release, comm, *routes});
    }
    /* Keep the provider loaded until process teardown: channel descriptors and
       forwarding leases must outlive the communicator's route cache. */
    std::printf("[A5 CCU URMA] provider installed %u explicit source route(s) from %s\n",
                routeCount, manifest);
    std::fflush(stdout);
    return HCCL_SUCCESS;
}

} // namespace a5_ccu_urma_probe
