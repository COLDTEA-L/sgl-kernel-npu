#include "a5_uvs_explicit_route_backend.h"
#include "a5_uvs_source_route_provider.h"

#include <dlfcn.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr uint32_t REQUIRED_CAPS = A5_UVS_CAP_INSTALL_SOURCE_ROUTE |
    A5_UVS_CAP_QUERY_SOURCE_ROUTE | A5_UVS_CAP_REMOVE_SOURCE_ROUTE |
    A5_UVS_CAP_IO_DIE_FORWARDING | A5_UVS_CAP_NO_RELAY_HBM;

struct ManifestRoute {
    A5UvsExplicitRouteRequest request{};
};

struct InstalledRoute {
    HcclComm comm = nullptr;
    uint32_t localRank = 0;
    uint32_t peerRank = 0;
    uint32_t routeId = 0;
    uint64_t leaseId = 0;
};

struct Backend {
    void *handle = nullptr;
    A5UvsBackendGetCapabilitiesFn getCapabilities = nullptr;
    A5UvsBackendInstallRouteFn installRoute = nullptr;
    A5UvsBackendQueryRouteFn queryRoute = nullptr;
    A5UvsBackendRemoveRouteFn removeRoute = nullptr;
    uint32_t maxRoutes = 0;
};

Backend g_backend;
std::mutex g_mutex;
std::vector<InstalledRoute> g_installedRoutes;

std::string Trim(const std::string &input)
{
    const size_t begin = input.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const size_t end = input.find_last_not_of(" \t\r\n");
    return input.substr(begin, end - begin + 1);
}

bool ParseUint32(const std::string &input, uint32_t *value)
{
    if (value == nullptr || input.empty()) {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(input.c_str(), &end, 10);
    if (errno != 0 || end == input.c_str() || *end != '\0' ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *value = static_cast<uint32_t>(parsed);
    return true;
}

std::vector<std::string> SplitCsv(const std::string &line)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
        fields.push_back(Trim(field));
    }
    return fields;
}

int ParseManifest(const char *path, uint32_t localRank, uint32_t peerRank,
                  std::vector<ManifestRoute> *routes)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        std::fprintf(stderr, "[A5 explicit relay] cannot open manifest: %s\n", path);
        return -1;
    }
    std::string line;
    uint32_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        line = Trim(line);
        if (line.empty() || line[0] == '#' || line.find("route_id") == 0) {
            continue;
        }
        const std::vector<std::string> fields = SplitCsv(line);
        if (fields.size() != 8) {
            std::fprintf(stderr,
                "[A5 explicit relay] manifest line %u requires 8 CSV fields\n", lineNumber);
            return -1;
        }
        uint32_t values[8] = {};
        for (size_t i = 0; i < fields.size(); ++i) {
            if (!ParseUint32(fields[i], &values[i])) {
                std::fprintf(stderr,
                    "[A5 explicit relay] invalid integer at manifest line %u field %zu\n",
                    lineNumber, i + 1);
                return -1;
            }
        }
        if (values[1] != localRank || values[2] != peerRank) {
            continue;
        }
        ManifestRoute route;
        route.request.routeId = values[0];
        route.request.localRank = values[1];
        route.request.peerRank = values[2];
        route.request.srcPhyId = values[3];
        route.request.dstPhyId = values[4];
        route.request.relayPhyId = values[5];
        route.request.dieId = values[6];
        route.request.weight = values[7] == 0 ? 1U : values[7];
        route.request.requireIoDieForwarding = 1;
        route.request.forbidRelayHbm = 1;
        if (route.request.relayPhyId == route.request.srcPhyId ||
            route.request.relayPhyId == route.request.dstPhyId) {
            std::fprintf(stderr,
                "[A5 explicit relay] route %u relay must differ from source and destination\n",
                route.request.routeId);
            return -1;
        }
        routes->push_back(route);
    }
    if (routes->empty()) {
        std::fprintf(stderr,
            "[A5 explicit relay] no manifest route for rank %u -> %u\n", localRank, peerRank);
        return -1;
    }
    return 0;
}

template <typename T>
bool LoadSymbol(void *handle, const char *name, T *symbol)
{
    *symbol = reinterpret_cast<T>(dlsym(handle, name));
    if (*symbol != nullptr) {
        return true;
    }
    std::fprintf(stderr, "[A5 explicit relay] backend lacks required symbol %s\n", name);
    return false;
}

int LoadBackend(Backend *backend)
{
    const char *path = std::getenv("A5_UVS_EXPLICIT_ROUTE_BACKEND");
    if (path == nullptr || path[0] == '\0') {
        std::fprintf(stderr,
            "[A5 explicit relay] A5_UVS_EXPLICIT_ROUTE_BACKEND is not set; "
            "plain URMA/uvs_get_route_list cannot install a selected relay\n");
        return -1;
    }
    backend->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (backend->handle == nullptr) {
        std::fprintf(stderr, "[A5 explicit relay] dlopen backend failed: %s\n", dlerror());
        return -1;
    }
    if (!LoadSymbol(backend->handle, "A5UvsBackendGetCapabilities", &backend->getCapabilities) ||
        !LoadSymbol(backend->handle, "A5UvsBackendInstallRoute", &backend->installRoute) ||
        !LoadSymbol(backend->handle, "A5UvsBackendQueryRoute", &backend->queryRoute) ||
        !LoadSymbol(backend->handle, "A5UvsBackendRemoveRoute", &backend->removeRoute)) {
        dlclose(backend->handle);
        *backend = Backend{};
        return -1;
    }
    A5UvsExplicitRouteBackendCaps caps{};
    const int status = backend->getCapabilities(
        A5_UVS_EXPLICIT_ROUTE_BACKEND_ABI_VERSION, &caps);
    if (status != 0 || caps.abiVersion != A5_UVS_EXPLICIT_ROUTE_BACKEND_ABI_VERSION ||
        (caps.capabilityFlags & REQUIRED_CAPS) != REQUIRED_CAPS) {
        std::fprintf(stderr,
            "[A5 explicit relay] backend capability gate failed: status=%d abi=%u "
            "flags=0x%x required=0x%x\n",
            status, caps.abiVersion, caps.capabilityFlags, REQUIRED_CAPS);
        dlclose(backend->handle);
        *backend = Backend{};
        return -1;
    }
    backend->maxRoutes = caps.maxRoutes;
    return 0;
}

void Rollback(const std::vector<uint64_t> &leases)
{
    for (auto it = leases.rbegin(); it != leases.rend(); ++it) {
        (void)g_backend.removeRoute(A5_UVS_EXPLICIT_ROUTE_BACKEND_ABI_VERSION, *it);
    }
}
} // namespace

extern "C" int A5UvsInstallSourceRoutes(uint32_t abiVersion, HcclComm comm,
    uint32_t localRank, uint32_t peerRank, const char *manifestPath,
    A5UvsSourceRoute *routes, uint32_t *routeCount)
{
    if (abiVersion != A5_UVS_SOURCE_ROUTE_ABI_VERSION || comm == nullptr ||
        manifestPath == nullptr || routes == nullptr || routeCount == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_backend.handle == nullptr && LoadBackend(&g_backend) != 0) {
        return -1;
    }
    std::vector<ManifestRoute> requested;
    if (ParseManifest(manifestPath, localRank, peerRank, &requested) != 0 ||
        requested.size() > *routeCount || requested.size() > g_backend.maxRoutes) {
        return -1;
    }

    std::vector<uint64_t> leases;
    for (size_t i = 0; i < requested.size(); ++i) {
        A5UvsExplicitRouteResult installed{};
        int status = g_backend.installRoute(A5_UVS_EXPLICIT_ROUTE_BACKEND_ABI_VERSION,
                                            comm, &requested[i].request, &installed);
        if (status != 0 || installed.leaseId == 0 || installed.forwardingOnly == 0 ||
            installed.installedSrcPhyId != requested[i].request.srcPhyId ||
            installed.installedDstPhyId != requested[i].request.dstPhyId ||
            installed.installedRelayPhyId != requested[i].request.relayPhyId ||
            installed.installedDieId != requested[i].request.dieId ||
            installed.channel.remoteRank != peerRank ||
            installed.channel.channelProtocol != COMM_PROTOCOL_UBC_CTP) {
            std::fprintf(stderr,
                "[A5 explicit relay] install/verification failed for route %u: "
                "status=%d lease=%lu relay=%u die=%u forwarding_only=%u\n",
                requested[i].request.routeId, status,
                static_cast<unsigned long>(installed.leaseId),
                installed.installedRelayPhyId, installed.installedDieId,
                installed.forwardingOnly);
            Rollback(leases);
            return -1;
        }
        A5UvsExplicitRouteResult queried{};
        status = g_backend.queryRoute(A5_UVS_EXPLICIT_ROUTE_BACKEND_ABI_VERSION,
                                      installed.leaseId, &queried);
        if (status != 0 || queried.installedSrcPhyId != requested[i].request.srcPhyId ||
            queried.installedDstPhyId != requested[i].request.dstPhyId ||
            queried.installedRelayPhyId != requested[i].request.relayPhyId ||
            queried.installedDieId != requested[i].request.dieId || queried.forwardingOnly == 0) {
            leases.push_back(installed.leaseId);
            Rollback(leases);
            return -1;
        }
        routes[i].channel = installed.channel;
        routes[i].routeId = requested[i].request.routeId;
        routes[i].relayPhyId = requested[i].request.relayPhyId;
        routes[i].dieId = requested[i].request.dieId;
        routes[i].weight = requested[i].request.weight;
        leases.push_back(installed.leaseId);
        g_installedRoutes.push_back(
            {comm, localRank, peerRank, requested[i].request.routeId, installed.leaseId});
        std::printf("[A5 explicit relay][rank=%u peer=%u] installed route=%u "
                    "relay_phy=%u die=%u lease=%lu forwarding_only=1 no_relay_hbm=1\n",
                    localRank, peerRank, requested[i].request.routeId,
                    requested[i].request.relayPhyId, requested[i].request.dieId,
                    static_cast<unsigned long>(installed.leaseId));
    }
    *routeCount = static_cast<uint32_t>(requested.size());
    return 0;
}

extern "C" void A5UvsReleaseSourceRoutes(HcclComm comm,
    const A5UvsSourceRoute *routes, uint32_t routeCount)
{
    if (comm == nullptr || routes == nullptr || g_backend.removeRoute == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    for (uint32_t i = 0; i < routeCount; ++i) {
        for (auto it = g_installedRoutes.begin(); it != g_installedRoutes.end();) {
            if (it->comm == comm && it->routeId == routes[i].routeId) {
                (void)g_backend.removeRoute(
                    A5_UVS_EXPLICIT_ROUTE_BACKEND_ABI_VERSION, it->leaseId);
                it = g_installedRoutes.erase(it);
            } else {
                ++it;
            }
        }
    }
}
