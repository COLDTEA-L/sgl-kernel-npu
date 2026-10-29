"""Public SDK path: no private 9.1 DFX registration, submit once and preserve errors."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[3]
class PublicDfxTests(unittest.TestCase):
    def test_sdk_owned_reporting_single_submission_and_failures(self):
        compiler = shutil.which('g++') or shutil.which('c++')
        if not compiler: self.skipTest('C++ compiler unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'hccl').mkdir()
            (root/'hccl/hccl_types.h').write_text('#pragma once\n#include <cstdint>\nusing HcclComm=void*; using HcclResult=int; constexpr int HCCL_SUCCESS=0,HCCL_E_PTR=2;\n')
            (root/'test.cc').write_text(r'''#include "ep8_dfx.h"
#include <cassert>
int main() {
    using a5_ccu_urma_probe::Ep8Dfx;
    std::shared_ptr<Ep8Dfx> dfx;
    assert(Ep8Dfx::Create(nullptr,1,&dfx)==2 && !dfx);
    assert(Ep8Dfx::Create(reinterpret_cast<void*>(1),1,&dfx)==0);
    int submits=0;
    for(int i=0;i<27000;++i)
        assert(dfx->Run(0,1,2,1,1,0,[&]{++submits;return 0;})==0);
    assert(submits==27000);
    assert(dfx->Run(0,1,2,1,1,0,[&]{++submits;return 2;})==2 && submits==27001);
}''')
            subprocess.run([compiler,'-std=c++14','-Wall','-Wextra','-Werror','-I',str(root),
                '-I',str(ROOT/'examples/a5_ccu_urma_route_probe/op_host'),str(root/'test.cc'),'-o',str(root/'test')],check=True)
            subprocess.run([str(root/'test')],check=True)
if __name__=='__main__': unittest.main()
