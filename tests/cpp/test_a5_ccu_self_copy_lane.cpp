// CPU asynchronous dependency model. Real ACL functions are replaced below;
// this does not prove a target device or graph backend supports these tasks.
#include "self_copy_lane.h"
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
using namespace a5_ccu_urma_probe;

namespace {
struct Task { std::vector<size_t> deps; std::function<void()> work; bool done = false; };
struct Stream { std::vector<size_t> tasks; };
struct Event { std::vector<size_t> progress; };
std::vector<Task> tasks;
int eventCreates = 0, eventDestroys = 0, streamCreates = 0, streamDestroys = 0;
int failEventCreate = 0;
bool failJoin = false;
Stream *mainStream = nullptr;
size_t lastCopy = 0;
uint32_t randomState = 1;

size_t Add(Stream *stream, std::function<void()> work, std::vector<size_t> deps = {})
{
    if (!stream->tasks.empty()) deps.push_back(stream->tasks.back());
    const size_t id = tasks.size();
    tasks.push_back({deps, work, false}); stream->tasks.push_back(id);
    return id;
}
bool Depends(size_t task, size_t dependency)
{
    if (task == dependency) return true;
    if (task < dependency) return false;
    for (size_t parent : tasks[task].deps) if (Depends(parent, dependency)) return true;
    return false;
}
void Drain(Stream *stream)
{
    while (!stream->tasks.empty() && !tasks[stream->tasks.back()].done) {
        std::vector<size_t> runnable;
        for (size_t i = 0; i < tasks.size(); ++i) {
            if (tasks[i].done) continue;
            bool ready = true;
            for (size_t dep : tasks[i].deps) ready &= tasks[dep].done;
            if (ready) runnable.push_back(i);
        }
        assert(!runnable.empty()); // a cyclic dependency is an immediate failure
        randomState = randomState * 1664525U + 1013904223U;
        const size_t id = runnable[randomState % runnable.size()];
        tasks[id].work(); tasks[id].done = true;
    }
}
unsigned char Value(uint32_t src, uint32_t dst, uint32_t epoch, size_t i)
{
    return static_cast<unsigned char>(src * 41 + dst * 17 + epoch * 13 + i % 251);
}
} // namespace

extern "C" aclError aclrtGetDevice(int32_t *device) { *device = 0; return ACL_SUCCESS; }
extern "C" aclError aclrtCreateStream(aclrtStream *stream)
{
    *stream = new Stream; ++streamCreates; return ACL_SUCCESS;
}
extern "C" aclError aclrtDestroyStream(aclrtStream stream)
{
    auto *s = static_cast<Stream *>(stream);
    assert(s->tasks.empty() || tasks[s->tasks.back()].done);
    delete s; ++streamDestroys; return ACL_SUCCESS;
}
extern "C" aclError aclrtCreateEventExWithFlag(aclrtEvent *event, uint32_t flags)
{
    assert(flags == ACL_EVENT_SYNC);
    if (failEventCreate && --failEventCreate == 0) return static_cast<aclError>(123);
    *event = new Event; ++eventCreates; return ACL_SUCCESS;
}
extern "C" aclError aclrtDestroyEvent(aclrtEvent event)
{
    auto *e = static_cast<Event *>(event);
    for (auto id : e->progress) assert(tasks[id].done);
    delete e; ++eventDestroys; return ACL_SUCCESS;
}
extern "C" aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream)
{
    auto *e = static_cast<Event *>(event);
    auto *s = static_cast<Stream *>(stream);
    const size_t id = Add(s, [] {});
    e->progress = {id}; return ACL_SUCCESS;
}
extern "C" aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event)
{
    if (failJoin && stream == mainStream) { failJoin = false; return static_cast<aclError>(124); }
    // Snapshot, not a reference to mutable event state. Mirrors Ex semantics.
    Add(static_cast<Stream *>(stream), [] {}, static_cast<Event *>(event)->progress);
    return ACL_SUCCESS;
}
extern "C" aclError aclrtMemcpyAsync(void *dst, size_t capacity, const void *src,
    size_t bytes, aclrtMemcpyKind kind, aclrtStream stream)
{
    assert(capacity >= bytes && kind == ACL_MEMCPY_DEVICE_TO_DEVICE);
    lastCopy = Add(static_cast<Stream *>(stream), [=] { std::memcpy(dst, src, bytes); });
    return ACL_SUCCESS;
}
extern "C" aclError aclrtSynchronizeStream(aclrtStream stream)
{
    Drain(static_cast<Stream *>(stream)); return ACL_SUCCESS;
}

void CheckQueuedCalls(uint32_t k, uint32_t rank, uint32_t elementBytes, bool overlap)
{
    tasks.clear();
    Stream main; mainStream = &main;
    setenv("A5_CCU_SELF_COPY_MODE", overlap ? "overlap" : "serial", 1);
    std::shared_ptr<SelfCopyLane> lane;
    const int eventsBefore = eventCreates, streamsBefore = streamCreates;
    assert(SelfCopyLane::Create(&main, &lane) == HCCL_SUCCESS);
    assert(lane->overlap() == overlap);
    const size_t bytes = 259 * elementBytes, total = k * bytes;
    std::vector<unsigned char> input(total + 32, 0xCD), output(total + 32, 0xCD);
    auto *send = input.data() + 16; auto *recv = output.data() + 16;
    int checked = 0;
    for (uint32_t epoch = 0; epoch < 50; ++epoch) {
        Add(&main, [=] {
            for (uint32_t dst = 0; dst < k; ++dst)
                for (size_t i = 0; i < bytes; ++i) send[dst * bytes + i] = Value(rank, dst, epoch, i);
            std::memset(recv, 0xA5, total);
        });
        SelfCopyGuard guard(lane);
        assert(guard.Begin(recv + rank * bytes, send + rank * bytes, bytes) == HCCL_SUCCESS);
        const size_t copyBefore = lastCopy;
        const size_t ccu = Add(&main, [=] {
            // A fake CCU checks its input producer and writes ONLY remote rows.
            for (uint32_t dst = 0; dst < k; ++dst)
                for (size_t i = 0; i < bytes; ++i) assert(send[dst * bytes + i] == Value(rank, dst, epoch, i));
            for (uint32_t src = 0; src < k; ++src) if (src != rank)
                for (size_t i = 0; i < bytes; ++i) recv[src * bytes + i] = Value(src, rank, epoch, i);
        });
        assert(guard.Finish(HCCL_SUCCESS) == HCCL_SUCCESS);
        const size_t copy = overlap ? lastCopy : copyBefore;
        assert(Depends(ccu, copy) == !overlap);
        if (overlap) assert(!Depends(copy, ccu));
        const size_t consumer = Add(&main, [&, epoch] {
            for (uint32_t src = 0; src < k; ++src)
                for (size_t i = 0; i < bytes; ++i) assert(recv[src * bytes + i] == Value(src, rank, epoch, i));
            for (size_t i = 0; i < 16; ++i) {
                assert(input[i] == 0xCD && input[total + 16 + i] == 0xCD);
                assert(output[i] == 0xCD && output[total + 16 + i] == 0xCD);
            }
            ++checked;
        });
        assert(Depends(consumer, ccu) && Depends(consumer, copy));
    }
    // No per-call synchronization: event records were overwritten 50 times.
    Drain(&main);
    assert(checked == 50);
    assert(eventCreates - eventsBefore == (overlap ? 2 : 0));
    assert(streamCreates - streamsBefore == (overlap ? 1 : 0));
    assert(lane->Close() == HCCL_SUCCESS);
    assert(lane->Close() == HCCL_SUCCESS);
    assert(lane->Begin(recv, send, bytes) != HCCL_SUCCESS);
}

void CheckFailures()
{
    tasks.clear(); Stream main; mainStream = &main;
    setenv("A5_CCU_SELF_COPY_MODE", "overlap", 1);
    std::shared_ptr<SelfCopyLane> lane;
    const int eventsBefore = eventCreates - eventDestroys;
    const int streamsBefore = streamCreates - streamDestroys;
    failEventCreate = 2;
    assert(SelfCopyLane::Create(&main, &lane) != HCCL_SUCCESS);
    assert(eventCreates - eventDestroys == eventsBefore);
    assert(streamCreates - streamDestroys == streamsBefore);
    assert(SelfCopyLane::Create(&main, &lane) == HCCL_SUCCESS);
    unsigned char send[32] = {}, recv[32] = {};
    const size_t before = tasks.size();
    {
        SelfCopyGuard guard(lane);
        assert(guard.Begin(recv, send, 32) == HCCL_SUCCESS);
        // RAII failure before CCU submission must NOT leave a detached copy.
    }
    assert(tasks.size() == before + 1); // input-ready record only
    assert(lane->Begin(recv, send, 32) != HCCL_SUCCESS);
    assert(lane->Close() == HCCL_SUCCESS);
    assert(SelfCopyLane::Create(&main, &lane) == HCCL_SUCCESS);
    {
        SelfCopyGuard guard(lane);
        assert(guard.Begin(recv, send, 32) == HCCL_SUCCESS);
        Add(&main, [] {}); // fake CCU submit succeeded
        failJoin = true;
        assert(guard.Finish(HCCL_SUCCESS) != HCCL_SUCCESS);
    }
    assert(lane->Begin(recv, send, 32) != HCCL_SUCCESS); // poisoned after API failure
    assert(lane->Close() == HCCL_SUCCESS);
    assert(eventCreates == eventDestroys && streamCreates == streamDestroys);
}

int main()
{
    for (bool overlap : {false, true}) for (uint32_t k : {2U, 4U})
        for (uint32_t rank = 0; rank < k; ++rank) for (uint32_t size : {1U, 2U, 4U})
            CheckQueuedCalls(k, rank, size, overlap);
    CheckFailures();
    std::cout << "copy/CCU dependency model, 50 queued epochs, guards, cleanup/failures: PASS\n";
}
