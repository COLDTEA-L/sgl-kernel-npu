"""Exercise the production DFX wrapper with delayed per-lane task reporting.

This models HCCL's copied op metadata and queue-full callbacks; it does not
claim to emulate the target vendor binary or to prove NPU long-run stability.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
HEADER = r'''#pragma once
#include <cstdint>
using HcclComm=void*; using HcclResult=int;
constexpr int HCCL_SUCCESS=0,HCCL_E_PARA=1,HCCL_E_PTR=2,HCCL_E_NOT_SUPPORT=6;
constexpr int HCCL_CMD_ALLTOALLV=8,HCCL_REDUCE_RESERVED=255,HCCL_DATA_TYPE_UINT8=7,COMM_ENGINE_CCU=5;
constexpr int COMM_NAME_MAX_LENGTH=128;
struct HcclDfxOpInfo {
    uint32_t opMode=0,opType=0,reduceOp=0,dataType=0;
    uint64_t dataCount=0; int engine=0; char algTag[288]{};
    MEMORY_FIELDS
};
'''
SOURCE = r'''
#include "ep8_dfx.h"
#include <cassert>
#include <deque>
#include <map>
#include <string>
#include <vector>
struct Op { void *comm=nullptr; HcclDfxOpInfo info{}; };
struct Lane { std::shared_ptr<Op> current=std::make_shared<Op>(); std::deque<std::shared_ptr<Op>> tasks; };
std::map<std::string,Lane> lanes;
int registered=0,launched=0,reports=0,regFailure=0,nameFailure=0;
extern "C" int HcclGetCommName(void *comm,char *name) {
    if(nameFailure) return nameFailure;
    std::snprintf(name,128,"lane_%llu",static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(comm)));
    return 0;
}
extern "C" int HcclDfxRegOpInfoByCommId(char *name,void *ptr) {
    if(regFailure) return regFailure;
    auto &lane=lanes[name]; auto info=*static_cast<HcclDfxOpInfo*>(ptr);
    assert(info.opMode==1 && info.opType==HCCL_CMD_ALLTOALLV && info.engine==COMM_ENGINE_CCU);
    assert(info.dataType==HCCL_DATA_TYPE_UINT8 && info.reduceOp==HCCL_REDUCE_RESERVED);
    assert(std::string(info.algTag).find("a5_ep8_alltoallv_")==0);
    lane.current=std::make_shared<Op>();lane.current->comm=&lane;lane.current->info=info;
    ++registered; return 0;
}
int launch(const std::string &name,uint64_t bytes) {
    ++launched;auto &lane=lanes[name]; lane.tasks.push_back(lane.current);
    assert(lane.current->info.dataCount==bytes);
    if(lane.tasks.size()==2048) {
        ++reports;
        for(auto &task:lane.tasks) if(!task->comm)return 2;
        lane.tasks.clear();
    }
    return 0;
}
int main() {
    // The unregistered path passes small runs, then fails in delayed reporting.
    for(int i=0;i<2047;++i)assert(launch("old",0)==0);
    assert(launch("old",0)==2);
    using a5_ccu_urma_probe::Ep8Dfx;
    std::vector<std::shared_ptr<Ep8Dfx>> dfx(3);
    for(uintptr_t i=0;i<3;++i)assert(Ep8Dfx::Create(reinterpret_cast<void*>(i+1),i+1,&dfx[i])==0);
    const int before=reports;
    for(int iteration=0;iteration<9000;++iteration) for(int lane=0;lane<3;++lane) {
        auto name="lane_"+std::to_string(lane+1);
        // A different native collective may replace the current metadata.
        // Prior queued tasks must keep their immutable, valid metadata.
        lanes[name].current=std::make_shared<Op>();
        uint64_t bytes=iteration%3 ? 0 : 8192;
        assert(dfx[lane]->Run(0,4096,8192,8192,8192,bytes,[&]{return launch(name,bytes);})==0);
    }
    assert(registered==27000 && reports-before==12);
    for(int i=1;i<=3;++i)for(auto &task:lanes["lane_"+std::to_string(i)].tasks)assert(task->comm);
    // Registration failure must not submit anything; preserve the first code.
    regFailure=42;int calls=0;
    assert(dfx[0]->Run(0,1,2,1,1,0,[&]{++calls;return 0;})==42 && calls==0);
    regFailure=0;
    assert(dfx[0]->Run(0,1,2,1,1,0,[&]{++calls;return 2;})==2 && calls==1);
    // Null comm/name lookup errors propagate before any registration or launch.
    std::shared_ptr<Ep8Dfx> invalid;
    assert(Ep8Dfx::Create(nullptr,1,&invalid)==2);
    nameFailure=43;
    assert(Ep8Dfx::Create(reinterpret_cast<void*>(1),1,&invalid)==43 && !invalid);
}
'''


class DfxTests(unittest.TestCase):
    def test_delayed_reporting_and_first_failure(self):
        compiler = shutil.which('g++') or shutil.which('c++')
        if not compiler: self.skipTest('C++ compiler unavailable')
        # Compile against both actual public-header shapes used by 9.1 SDKs.
        for fields in ('int8_t reserve[128]{};',
                       'uint64_t inputMemAddr=0,outputMemAddr=0,inputMemSize=0,outputMemSize=0; int8_t reserve[96]{};'):
            with self.subTest(fields=fields), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'hccl').mkdir()
                (root / 'hccl/hccl_diag.h').write_text(HEADER.replace('MEMORY_FIELDS', fields))
                (root / 'test.cc').write_text(SOURCE)
                command = [compiler, '-std=c++14', '-Wall', '-Wextra', '-Werror', '-rdynamic',
                           '-I', str(root), '-I', str(ROOT / 'examples/a5_ccu_urma_route_probe/op_host'),
                           str(root / 'test.cc'), '-ldl', '-o', str(root / 'test')]
                subprocess.run(command, check=True, capture_output=True, text=True)
                subprocess.run([str(root / 'test')], check=True, capture_output=True, text=True)


if __name__ == '__main__': unittest.main()
