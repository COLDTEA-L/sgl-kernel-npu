// CPU-only protocol worker. It deliberately simulates the graph worker's log
// protocol for launcher integration tests; it never initializes an NPU.
#include "../../examples/a5_ccu_urma_route_probe/testcase/graph_gate_scheduler.h"
#include <cstdio>
#include <thread>

int main(int argc, char **argv)
{
    try {
        if (argc != 10) throw std::runtime_error("CPU protocol fixture expects graph worker arguments");
        std::puts("CPU_PROTOCOL_ONLY_NO_DEVICE_EXECUTION");
        const unsigned rank = std::stoul(argv[1]), iterations = std::stoul(argv[5]);
        const size_t bytes = std::stoull(argv[4]);
        GateSchedulerClient client(rank);
        if (!client.Enabled()) throw std::runtime_error("CPU protocol fixture requires a scheduler");
        for (unsigned it = 0; it < iterations; ++it) {
            for (unsigned op = 0; op < 2; ++op) {
                client.Arrive(it, op, 3, bytes);
                std::this_thread::sleep_for(std::chrono::milliseconds(rank == 0 ? 1 : 5));
                client.Done(it, op, 1000000);
                std::printf("GATE_OP_PASS rank=%u iteration=%u op=%u\n", rank, it, op);
                std::fflush(stdout);
            }
        }
        std::printf("GRAPH_GATE_RANK_PASS rank=%u iterations=%u\n", rank, iterations);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "CPU_PROTOCOL_FAIL %s\n", error.what());
        return 1;
    }
}
