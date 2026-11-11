// CPU planning only, not communication latency. Can compile against an old header
// using -I /path/to/old/header to make a same-host before/after comparison.
#include "ep8_relay_planner.hpp"
#include <chrono>
#include <iostream>
using namespace deep_ep::ep8;
int main() {
    std::vector<std::vector<U>> allowed(64), shared(64), dedicated(64);
    for(size_t s=0;s<8;++s)for(size_t d=0;d<8;++d)if(s!=d) {
        shared[s*8+d]={s*18+1,d*18+1};dedicated[s*8+d]={s*18+d,d*18+s};
        for(size_t k=0;k<8;++k)if(k!=s && k!=d)allowed[s*8+d].push_back(k);
    }
    Matrix real{
        2899,398,419,792,647,606,1197,1670,
        2072,440,448,583,1192,833,1368,1680,
        3027,441,583,664,587,438,1329,1559,
        1946,405,537,787,1191,819,1071,1860,
        2616,522,479,648,520,800,1006,1545,
        1965,413,449,880,1261,802,976,1870,
        2913,706,394,465,580,402,1174,1748,
        1765,442,312,1047,1253,887,951,1959};
    Matrix uniform;uniform.fill(3072);
    volatile U checksum=0;
    for(U limit:{8u,16u,56u}) for(bool separate:{false,true}) for(bool balanced:{false,true}) {
        auto planner=Planner(allowed,5120,1,0,limit,{}).WithL1Routes(separate?dedicated:shared);
        auto c=balanced?uniform:real;Plan plan;
        for(int i=0;i<100;++i)planner.MakeInto(c,plan);
        std::array<double,5> samples{};
        for(auto& us:samples) {
            auto start=std::chrono::steady_clock::now();
            for(int i=0;i<10000;++i) {planner.MakeInto(c,plan);checksum=checksum+plan.moved[1];}
            us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/10000;
        }
        std::sort(samples.begin(),samples.end());
        std::cout<<"L1_PLAN_CPU matrix="<<(balanced?"uniform":"matrix_few")
                 <<" ports="<<(separate?"per_peer":"shared")<<" max_actions="<<limit
                 <<" actions="<<plan.actions.size()<<" evaluations="<<plan.evaluations
                 <<" median_us="<<samples[2]<<"\n";
    }
    return checksum==Max;
}
