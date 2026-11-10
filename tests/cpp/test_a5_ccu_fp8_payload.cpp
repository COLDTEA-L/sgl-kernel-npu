// Real ATen and production byte-copy/pipeline host. No NPU or collective emulation.
#include "../../csrc/deepep/ep8_pipeline_host.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace deep_ep;
using namespace deep_ep::ep8;
static void Require(bool value) { if (!value) throw std::runtime_error("FP8 CPU check failed"); }
struct HostRuntime : PipelineRuntime {
    size_t transfers = 0;
    void Check(const at::Tensor& t) override { Require(t.device().is_cpu()); }
    void Sync() override {}
    void Lane(int) override {}
    void Events(size_t) override {}
    void Record(size_t, int) override {}
    void Wait(size_t, int) override {}
    void Retain(const at::Tensor&, int) override {}
    void Copy(const at::Tensor& src, const at::Tensor& dst, const CopySpans& spans) override { CopySpansOut(src, dst, spans); }
    void Transfer(const at::Tensor&, const at::Tensor&, const Matrix&, const Matrix*, U) override { ++transfers; }
};
int main() {
    std::vector<std::vector<U>> allowed(64);
    for (int s = 0; s < 8; ++s) for (int d = 0; d < 8; ++d) if (s != d)
        for (U k = 0; k < 8; ++k) if (k != static_cast<U>(s) && k != static_cast<U>(d)) allowed[s*8+d].push_back(k);
    size_t submissions = 0;
    for (auto type : {at::kFloat8_e4m3fn, at::kFloat8_e5m2, at::kFloat8_e4m3fnuz, at::kFloat8_e5m2fnuz}) {
        Require(IsCcuFp8(type) && IsCcuPayload(type) && !IsCcuPayload(at::kDouble));
        constexpr int width = 257, guard = 37, rows = 20;
        auto srcRaw = at::empty({rows*width + 2*guard}, at::TensorOptions().dtype(at::kByte));
        auto dstRaw = at::full({rows*width + 2*guard}, 239, srcRaw.options());
        auto bytes = srcRaw.data_ptr<uint8_t>();
        for (int i = 0; i < srcRaw.numel(); ++i) bytes[i] = i % 256;
        auto src = srcRaw.narrow(0, guard, rows*width).view({rows, width}).view(type);
        auto dst = dstRaw.narrow(0, guard, rows*width).view({rows, width}).view(type);
        auto expected = dstRaw.clone();
        CopySpans spans({{0, 1, 3}, {7, 5, 4}, {19, 16, 1}, {20, 20, 0}});
        for (const auto& span : spans.spans)
            std::memcpy(expected.data_ptr<uint8_t>() + guard + span[1]*width,
                        bytes + guard + span[0]*width, span[2]*width);
        CopySpansOut(src, dst, spans);
        Require(at::equal(expected, dstRaw));
        Require(CcuByteView(src).data_ptr() == src.data_ptr() && CcuByteView(src).storage_offset() == guard);
        auto before = dstRaw.clone();
        bool rejected = false;
        try { CopySpansOut(src, dst, CopySpans({{0, 0, 1}, {20, 0, 1}})); }
        catch (const c10::Error&) { rejected = true; }
        Require(rejected && at::equal(before, dstRaw));
        CopySpansOut(src.narrow(0, 0, 0), dst.narrow(0, 0, 0), CopySpans());

        Planner planner(allowed, width, 1, 0, 56, {});
        auto runtime = std::make_shared<HostRuntime>();
        PipelineHost host(planner, 0, 7*width, 128, 1<<20, runtime, false);
        Matrix counts{}; counts[0] = 3; counts[4] = 65;
        auto send = at::empty({68, width}, src.options());
        auto recv = at::empty({3, width}, src.options());
        auto prepared = host.Prepare(counts, send);
        Require(prepared->UsesRelay());
        auto allocations = host.Stats()[1];
        for (int step = 0; step < 100; ++step) host.Run(counts, send, recv);
        Require(host.Stats()[1] == allocations);
        submissions += runtime->transfers;
        host.Close();
    }
    std::cout << "FP8_CPP_PAYLOAD_PASS formats=4 pipeline_runs=400 submissions=" << submissions << "\n";
}
