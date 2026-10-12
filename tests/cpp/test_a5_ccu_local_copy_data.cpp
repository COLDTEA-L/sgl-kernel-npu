#include "examples/a5_ccu_urma_route_probe/testcase/local_copy_probe_data.h"
#include <cassert>
#include <cstdio>
#include <vector>
using namespace a5_local_probe;
int main()
{
    assert(ValidSize(4096, 20, false)); assert(ValidSize(4194304, 20, true));
    assert(!ValidSize(256, 20, true)); assert(!ValidSize(0, 20, false));
    assert(!ValidSize(MAX_BYTES, 20, false)); assert(!ValidSize(4096, 0, false));
    assert(!ValidSize(4096, 33, false));
    for (size_t size : {256U, 4096U, 65536U}) {
        std::vector<uint8_t> input(size + 2 * GUARD), output(input.size());
        for (uint32_t rank = 0; rank < 2; ++rank) for (uint32_t epoch = 0; epoch < 30; ++epoch) {
            Fill(input.data(), size, rank, epoch, true);
            Fill(output.data(), size, rank, epoch, false);
            assert(Verify(input.data(), size, rank, epoch, true, nullptr));
            assert(!Verify(output.data(), size, rank, epoch, false, nullptr));
            std::memcpy(output.data() + GUARD, input.data() + GUARD, size);
            assert(Verify(output.data(), size, rank, epoch, false, nullptr));
            for (size_t index : {size_t(0), GUARD + size / 2, output.size() - 1}) {
                output[index] ^= 1; size_t bad = 0;
                assert(!Verify(output.data(), size, rank, epoch, false, &bad) && bad == index);
                output[index] ^= 1;
            }
        }
    }
    std::puts("local-copy pattern/poison/canary model: PASS (CPU only)");
}
