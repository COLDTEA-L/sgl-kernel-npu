#pragma once
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace l0l1 {
// Caller supplies SDK enum comparisons; only UB_CTP + EID/IPv6 is admitted.
inline bool UbEidAddress(bool ubCtp, bool eidOrV6, const uint8_t* bytes) {
    if (!ubCtp || !eidOrV6) return false;
    for (unsigned i=0; i<16; ++i) if (bytes[i]) return true;
    return false;
}
struct EndpointAttrs {
    int dieStatus=-1; uint32_t die=0;
    int bwStatus=-1; uint32_t bw=0;
    bool operator==(const EndpointAttrs& other) const {
        return dieStatus==other.dieStatus && die==other.die && bwStatus==other.bwStatus && bw==other.bw;
    }
};
using EndpointTable=std::map<std::string,EndpointAttrs>;
inline void AddEndpoint(EndpointTable& table,const std::string& key,const EndpointAttrs& value) {
    auto inserted=table.emplace(key,value);
    if(!inserted.second && !(inserted.first->second==value))
        throw std::runtime_error("ENDPOINT_LOCAL_ATTRIBUTE_CONFLICT key="+key);
}
inline std::string EncodeEndpoints(unsigned rank,const EndpointTable& table) {
    std::ostringstream out; out<<"L0L1_ENDPOINT_ATTRS_V1 "<<rank<<' '<<table.size()<<'\n';
    for(const auto& item:table) {
        const auto& a=item.second;
        out<<item.first<<' '<<a.dieStatus<<' '<<a.die<<' '<<a.bwStatus<<' '<<a.bw<<'\n';
    }
    return out.str();
}
inline EndpointTable DecodeEndpoints(const std::string& data,unsigned expectedRank) {
    std::istringstream in(data); std::string tag; unsigned rank=2; size_t count=0;
    if(!(in>>tag>>rank>>count) || tag!="L0L1_ENDPOINT_ATTRS_V1" || rank!=expectedRank || count>4096)
        throw std::runtime_error("invalid peer endpoint attribute header");
    EndpointTable result;
    for(size_t i=0;i<count;++i) {
        std::string key; EndpointAttrs a;
        if(!(in>>key>>a.dieStatus>>a.die>>a.bwStatus>>a.bw) ||
           (a.dieStatus==0 && a.die>1) || !result.emplace(key,a).second)
            throw std::runtime_error("invalid/duplicate peer endpoint attributes");
    }
    std::string extra;
    if(in>>extra) throw std::runtime_error("trailing peer endpoint attributes");
    return result;
}
inline EndpointAttrs PeerEndpoint(const EndpointTable& peer,const std::string& key,const EndpointAttrs& graph) {
    auto it=peer.find(key);
    if(it==peer.end()) throw std::runtime_error("PEER_ENDPOINT_MISSING key="+key);
    const auto& a=it->second;
    if((graph.dieStatus==0 && a.dieStatus==0 && graph.die!=a.die) ||
       (graph.bwStatus==0 && a.bwStatus==0 && graph.bw!=a.bw))
        throw std::runtime_error("ENDPOINT_GRAPH_PEER_CONFLICT key="+key);
    // Peer reports its LOCAL lookup, including failures. Never invent bandwidth.
    return a;
}
} // namespace l0l1
