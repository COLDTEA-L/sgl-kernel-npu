"""Compile the production EP8 event bridge against an asynchronous ACL model."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class ThreadBridgeTests(unittest.TestCase):
    def test_persistent_events_queued_launches_and_failed_record(self):
        compiler = shutil.which('g++') or shutil.which('c++')
        if not compiler: self.skipTest('C++ compiler unavailable')
        acl = '''#pragma once
#include <cstdint>
using aclError=int; using aclrtStream=void*; using aclrtEvent=void*;
constexpr int ACL_SUCCESS=0; constexpr uint32_t ACL_EVENT_SYNC=1;
extern "C" {
int aclrtGetDevice(int32_t*);
int aclrtRecordEvent(aclrtEvent,aclrtStream);
int aclrtStreamWaitEvent(aclrtStream,aclrtEvent);
int aclrtDestroyEvent(aclrtEvent);
int aclrtSynchronizeStream(aclrtStream);
}
'''
        hccl = '''#pragma once
using HcclResult=int;
constexpr int HCCL_SUCCESS=0,HCCL_E_PARA=1,HCCL_E_PTR=2,HCCL_E_NOT_SUPPORT=3,HCCL_E_RUNTIME=4;
'''
        source = r'''
#include "thread_event_bridge.h"
#include <cassert>
#include <functional>
#include <vector>
struct Node { std::vector<int> deps; std::function<void()> fn; bool done=false; };
struct Stream { int tail=-1; };
struct Event { int record=-1; };
std::vector<Node> nodes;
int creates=0,destroys=0,waits=0; bool failRecord=false;
int append(Stream *s, std::function<void()> fn={}, int extra=-1) {
    Node n; n.fn=fn;
    if(s->tail>=0)n.deps.push_back(s->tail);
    if(extra>=0)n.deps.push_back(extra);
    nodes.push_back(n); return s->tail=static_cast<int>(nodes.size())-1;
}
void drain() {
    size_t left=0;for(auto &n:nodes)left+=!n.done;
    while(left) {
        bool progress=false;
        for(size_t j=nodes.size();j>0;--j) {
            auto &n=nodes[j-1]; if(n.done)continue;
            bool ready=true;for(int dep:n.deps)ready &= nodes[dep].done;
            if(!ready)continue;
            if(n.fn)n.fn();n.done=true;--left;progress=true;
        }
        assert(progress);
    }
}
extern "C" int aclrtCreateEventExWithFlag(aclrtEvent *p,uint32_t f) {
    assert(f==ACL_EVENT_SYNC);*p=new Event;++creates;return 0;
}
extern "C" int aclrtGetDevice(int32_t *p){*p=0;return 0;}
extern "C" int aclrtRecordEvent(aclrtEvent e,aclrtStream s) {
    if(failRecord)return 42;
    static_cast<Event*>(e)->record=append(static_cast<Stream*>(s));return 0;
}
extern "C" int aclrtStreamWaitEvent(aclrtStream s,aclrtEvent e) {
    int record=static_cast<Event*>(e)->record; assert(record>=0);
    // Capture this record generation, not the event's later mutable state.
    append(static_cast<Stream*>(s),{},record);++waits;return 0;
}
extern "C" int aclrtDestroyEvent(aclrtEvent e){delete static_cast<Event*>(e);++destroys;return 0;}
extern "C" int aclrtSynchronizeStream(aclrtStream){drain();return 0;}
int main() {
    using a5_ccu_urma_probe::ThreadEventBridge;
    Stream main,route;std::shared_ptr<ThreadEventBridge> bridge;
    assert(ThreadEventBridge::Create(&main,&route,&bridge)==0);
    int input=-1,output=-1,consumed=0;
    for(int epoch=0;epoch<1024;++epoch) {
        append(&main,[&,epoch]{input=epoch;});
        assert(bridge->Begin()==0);
        append(&route,[&,epoch]{assert(input==epoch);output=input;});
        assert(bridge->Finish(0)==0);
        append(&main,[&,epoch]{assert(output==epoch);++consumed;});
    }
    assert(creates==2 && waits==2048 && consumed==0);
    assert(bridge->Close()==0 && consumed==1024 && destroys==2);
    assert(bridge->Close()==0 && destroys==2);
    assert(bridge->Begin()!=0);
    nodes.clear(); main.tail=route.tail=-1;
    assert(ThreadEventBridge::Create(&main,&route,&bridge)==0);
    assert(bridge->Begin()==0); int previous=waits;
    failRecord=true;
    assert(bridge->Finish(0)==HCCL_E_RUNTIME);
    assert(waits==previous); // No stranded caller wait after record failed.
    assert(bridge->Begin()!=0); // Poisoned bridge cannot retry a partial launch.
    failRecord=false;
    assert(bridge->Close()==0 && creates==4 && destroys==4);
}
'''
        with tempfile.TemporaryDirectory(prefix='a5-bridge-') as temp:
            tmp = Path(temp)
            (tmp / 'acl').mkdir(); (tmp / 'hccl').mkdir()
            (tmp / 'acl/acl_rt.h').write_text(acl)
            (tmp / 'hccl/hccl_types.h').write_text(hccl)
            (tmp / 'test.cc').write_text(source)
            exe = tmp / 'test'
            subprocess.run([compiler, '-std=c++14', '-rdynamic', '-I' + str(tmp),
                            '-I' + str(ROOT / 'examples/a5_ccu_urma_route_probe/op_host'),
                            str(tmp / 'test.cc'), '-ldl', '-o', str(exe)], check=True, capture_output=True, text=True)
            subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=20)


if __name__ == '__main__': unittest.main()
