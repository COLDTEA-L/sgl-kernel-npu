#include <acl/acl.h>
#include <hccl/hccl_channel.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=3) {std::fprintf(stderr,"usage: l1_inventory LOGICAL_DEVICE PHYSICAL_DEVICE\n");return 2;}
    const int logical=std::atoi(argv[1]),physical=std::atoi(argv[2]);
    if(aclInit(nullptr)!=ACL_SUCCESS || aclrtSetDevice(logical)!=ACL_SUCCESS) return 3;
    void* lib=dlopen("libhcomm.so",RTLD_NOW|RTLD_GLOBAL);
    using Num=int(*)(int32_t,uint32_t*);
    using Descs=int(*)(int32_t,uint32_t*,EndpointDesc*);
    auto num=reinterpret_cast<Num>(lib?dlsym(lib,"HcommEndpointGetDescNum"):nullptr);
    auto get=reinterpret_cast<Descs>(lib?dlsym(lib,"HcommEndpointGetDescs"):nullptr);
    uint32_t n=0;
    if(!num || !get || num(logical,&n)!=0 || !n || n>4096) return 4;
    std::vector<EndpointDesc> eps(n);const auto cap=n;
    if(get(logical,&n,eps.data())!=0 || n>cap) return 5;
    for(uint32_t i=0;i<n;++i) {
        const auto& e=eps[i];
        if(e.loc.locType!=ENDPOINT_LOC_TYPE_DEVICE || e.protocol!=COMM_PROTOCOL_UBC_CTP ||
           (e.commAddr.type!=COMM_ADDR_TYPE_EID && e.commAddr.type!=COMM_ADDR_TYPE_IP_V6)) continue;
        if(e.loc.device.devPhyId!=static_cast<uint32_t>(physical)) {
            std::fprintf(stderr,"endpoint physical mapping mismatch\n");return 6;
        }
        std::printf("L1_INVENTORY_ENDPOINT {\"device_id\":%d,\"protocol\":%d,\"addr_type\":%d,\"eid\":\"",physical,int(e.protocol),int(e.commAddr.type));
        for(auto x:e.commAddr.eid) std::printf("%02x",unsigned(x));
        std::printf("\"}\n");
    }
    std::fflush(stdout);
    if(aclrtResetDevice(logical)!=ACL_SUCCESS) return 7;
    return aclFinalize()==ACL_SUCCESS?0:8;
}
