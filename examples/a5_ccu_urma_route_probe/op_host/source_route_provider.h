#ifndef A5_SOURCE_ROUTE_PROVIDER_LOADER_H
#define A5_SOURCE_ROUTE_PROVIDER_LOADER_H

#include "a5_uvs_source_route_provider.h"

#include <vector>

namespace a5_ccu_urma_probe {

HcclResult LoadProviderRoutes(HcclComm comm, uint32_t rank, uint32_t peer,
                              std::vector<A5UvsSourceRoute> *routes);

} // namespace a5_ccu_urma_probe

#endif
