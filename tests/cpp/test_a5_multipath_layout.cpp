#include "examples/a5_ccu_urma_route_probe/common/path_layout.h"
#include <cassert>
#include <iostream>

using namespace a5_ccu_urma_probe;

static void CheckCoverage(uint64_t elements, uint32_t rank,
                          const std::vector<uint32_t> &weights)
{
    TwoRankPathLayout layout;
    assert(BuildTwoRankPathLayout(elements, rank, weights, &layout));
    uint64_t cursor = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        assert(layout.sourceOffsets[i] == (1 - rank) * layout.peerBytes + cursor);
        assert(layout.remoteOffsets[i] == rank * layout.peerBytes + cursor);
        assert(layout.pathBytes[i] > 0);
        assert(layout.sourceOffsets[i] + layout.pathBytes[i] <= layout.totalBytes);
        assert(layout.remoteOffsets[i] + layout.pathBytes[i] <= layout.totalBytes);
        cursor += layout.pathBytes[i];
    }
    assert(cursor == layout.peerBytes);
}

int main()
{
    for (uint32_t rank : {0U, 1U}) {
        CheckCoverage(1048576, rank, {1}); // native0/native2: one candidate
        CheckCoverage(1, rank, {1}); // single path has no aligned split
        CheckCoverage(1048577, rank, {UINT32_MAX});
        for (size_t relays = 1; relays <= 12; ++relays) {
            std::vector<uint32_t> weights(relays + 1, 1);
            weights[0] = 2;
            CheckCoverage(1048576, rank, weights);
        }
        CheckCoverage(1048577, rank, {2, 1, 1}); // residual belongs to last path
        CheckCoverage(1ULL << 33, rank, {UINT32_MAX, UINT32_MAX}); // overflowed old product
    }
    TwoRankPathLayout layout;
    assert(!BuildTwoRankPathLayout(1024, 0, {}, &layout));
    assert(!BuildTwoRankPathLayout(1, 0, {2, 1}, &layout));
    assert(!BuildTwoRankPathLayout(1024, 0, {0, 1}, &layout));
    assert(!BuildTwoRankPathLayout(UINT64_MAX, 0, {2, 1}, &layout));
    assert(!BuildTwoRankPathLayout(1024, 2, {2, 1}, &layout));
    assert(!BuildTwoRankPathLayout(1024, 0, std::vector<uint32_t>(14, 1), &layout));
    assert(DisjointBufferRanges(4096, 8192, 4096));
    assert(!DisjointBufferRanges(4096, 8191, 4096));
    assert(!DisjointBufferRanges(4096, 4096, 4096));
    assert(!DisjointBufferRanges(UINT64_MAX - 10, 4096, 64));
    assert(MakePathKernelSignature("route", {1, 23}, std::vector<uint64_t>{7, 8}) !=
           MakePathKernelSignature("route", {12, 3}, std::vector<uint64_t>{7, 8}));
    assert(MakePathKernelSignature("route", {0, 1000}, std::vector<uint64_t>{7, 8}) !=
           MakePathKernelSignature("route", {0, 1000}, std::vector<uint64_t>{7, 9}));
    std::cout << "layout, overflow, alias and signature checks PASS\n";
}
