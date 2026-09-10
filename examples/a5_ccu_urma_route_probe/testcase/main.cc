/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <acl/acl_rt.h>
#include <hccl/hccl_types.h>
#include <a5_ccu_urma_route_probe.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

struct Options {
    uint64_t bytes = 2ULL * 1024ULL * 1024ULL;
    uint32_t warmup = 10;
    uint32_t iterations = 100;
    uint32_t routeIndex = 0;
    bool remoteOnly = false;
    bool channelOnly = false;
};

struct ThreadContext {
    HcclRootInfo *rootInfo = nullptr;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    const Options *options = nullptr;
    std::atomic<int> *failed = nullptr;
};

static void PrintUsage(const char *program)
{
    std::cout << "Usage: " << program
              << " [--bytes N] [--warmup N] [--iters N] [--route-index N]"
              << " [--remote-only] [--channel-only]" << std::endl;
}

static bool ParseUint(const char *value, uint64_t *result)
{
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        return false;
    }
    *result = static_cast<uint64_t>(parsed);
    return true;
}

static bool ParseOptions(int argc, char **argv, Options *options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string name(argv[i]);
        if (name == "--remote-only") {
            options->remoteOnly = true;
            continue;
        }
        if (name == "--channel-only") {
            options->channelOnly = true;
            continue;
        }
        if (i + 1 >= argc) {
            return false;
        }
        uint64_t value = 0;
        if (!ParseUint(argv[++i], &value)) {
            return false;
        }
        if (name == "--bytes") {
            options->bytes = value;
        } else if (name == "--warmup") {
            options->warmup = static_cast<uint32_t>(value);
        } else if (name == "--iters") {
            options->iterations = static_cast<uint32_t>(value);
        } else if (name == "--route-index") {
            options->routeIndex = static_cast<uint32_t>(value);
        } else {
            return false;
        }
    }
    return options->bytes != 0 && options->bytes % sizeof(float) == 0 &&
           options->iterations != 0 && !(options->remoteOnly && options->channelOnly);
}

static double Percentile(const std::vector<double> &sorted, double percentile)
{
    if (sorted.empty()) {
        return 0.0;
    }
    const size_t index = static_cast<size_t>(percentile * static_cast<double>(sorted.size() - 1));
    return sorted[index];
}

#define THREAD_ACL_CHECK(expression)                                                        \
    do {                                                                                    \
        aclError status = (expression);                                                     \
        if (status != ACL_SUCCESS) {                                                        \
            std::cerr << "[rank=" << ctx->rank << "] ACL failure at " << __FILE__ << ':' \
                      << __LINE__ << ", status=" << status << std::endl;                   \
            ctx->failed->store(1);                                                         \
            return;                                                                         \
        }                                                                                   \
    } while (0)

#define THREAD_HCCL_CHECK(expression)                                                        \
    do {                                                                                     \
        HcclResult status = (expression);                                                    \
        if (status != HCCL_SUCCESS) {                                                        \
            std::cerr << "[rank=" << ctx->rank << "] HCCL failure at " << __FILE__ << ':' \
                      << __LINE__ << ", status=" << status << std::endl;                    \
            ctx->failed->store(1);                                                          \
            return;                                                                          \
        }                                                                                    \
    } while (0)

static void RunRank(ThreadContext *ctx)
{
    const uint64_t sendCount = ctx->options->bytes / sizeof(float);
    const uint64_t recvBytes = ctx->options->bytes * ctx->rankSize;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    aclrtStream stream = nullptr;
    HcclComm comm = nullptr;

    THREAD_ACL_CHECK(aclrtSetDevice(static_cast<int32_t>(ctx->rank)));
    THREAD_HCCL_CHECK(HcclCommInitRootInfo(ctx->rankSize, ctx->rootInfo, ctx->rank, &comm));
    THREAD_ACL_CHECK(aclrtCreateStream(&stream));
    THREAD_ACL_CHECK(aclrtMalloc(&sendBuf, ctx->options->bytes, ACL_MEM_MALLOC_HUGE_ONLY));
    THREAD_ACL_CHECK(aclrtMalloc(&recvBuf, recvBytes, ACL_MEM_MALLOC_HUGE_ONLY));

    std::vector<float> input(sendCount, static_cast<float>(ctx->rank + 1));
    THREAD_ACL_CHECK(aclrtMemcpy(sendBuf, ctx->options->bytes, input.data(), ctx->options->bytes,
                                ACL_MEMCPY_HOST_TO_DEVICE));
    if (ctx->options->remoteOnly) {
        auto *localDst = static_cast<uint8_t *>(recvBuf) + ctx->rank * ctx->options->bytes;
        THREAD_ACL_CHECK(aclrtMemcpy(localDst, ctx->options->bytes, sendBuf, ctx->options->bytes,
                                    ACL_MEMCPY_DEVICE_TO_DEVICE));
    }

    for (uint32_t i = 0; i < ctx->options->warmup; ++i) {
        THREAD_HCCL_CHECK(HcclCcuUrmaRouteProbe(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, comm, stream));
        THREAD_ACL_CHECK(aclrtSynchronizeStream(stream));
    }

    std::vector<double> samplesUs;
    samplesUs.reserve(ctx->options->iterations);
    for (uint32_t i = 0; i < ctx->options->iterations; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        THREAD_HCCL_CHECK(HcclCcuUrmaRouteProbe(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, comm, stream));
        THREAD_ACL_CHECK(aclrtSynchronizeStream(stream));
        const auto end = std::chrono::steady_clock::now();
        samplesUs.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    }

    bool verified = true;
    if (!ctx->options->channelOnly) {
        std::vector<float> output(recvBytes / sizeof(float));
        THREAD_ACL_CHECK(aclrtMemcpy(output.data(), recvBytes, recvBuf, recvBytes, ACL_MEMCPY_DEVICE_TO_HOST));
        for (uint32_t sourceRank = 0; sourceRank < ctx->rankSize && verified; ++sourceRank) {
            const float expected = static_cast<float>(sourceRank + 1);
            const uint64_t offset = sourceRank * sendCount;
            for (uint64_t i = 0; i < sendCount; ++i) {
                if (std::fabs(output[offset + i] - expected) > 1e-6F) {
                    std::cerr << "[rank=" << ctx->rank << "] mismatch at source_rank=" << sourceRank
                              << " element=" << i << " expected=" << expected
                              << " actual=" << output[offset + i] << std::endl;
                    ctx->failed->store(1);
                    verified = false;
                    break;
                }
            }
        }
    }

    if (verified) {
        std::sort(samplesUs.begin(), samplesUs.end());
        const double totalUs = std::accumulate(samplesUs.begin(), samplesUs.end(), 0.0);
        const double avgUs = totalUs / static_cast<double>(samplesUs.size());
        const double effectiveGbps = ctx->options->channelOnly ? 0.0 :
            static_cast<double>(ctx->options->bytes) * 8.0 / (avgUs * 1000.0);
        std::cout << "[rank=" << ctx->rank << "] PASS engine=CCU protocol=UBC_CTP route="
                  << ctx->options->routeIndex << " bytes_per_rank=" << ctx->options->bytes
                  << " warmup=" << ctx->options->warmup << " iterations=" << ctx->options->iterations
                  << " mode=" << (ctx->options->channelOnly ? "channel-only" :
                                    (ctx->options->remoteOnly ? "remote-only" : "allgather"))
                  << " avg_us=" << avgUs
                  << " min_us=" << samplesUs.front()
                  << " p50_us=" << Percentile(samplesUs, 0.50)
                  << " p95_us=" << Percentile(samplesUs, 0.95);
        if (!ctx->options->channelOnly) {
            std::cout << " effective_gbps=" << effectiveGbps;
        }
        std::cout << std::endl;
    }

    THREAD_HCCL_CHECK(HcclCommDestroy(comm));
    THREAD_ACL_CHECK(aclrtFree(sendBuf));
    THREAD_ACL_CHECK(aclrtFree(recvBuf));
    THREAD_ACL_CHECK(aclrtDestroyStream(stream));
    THREAD_ACL_CHECK(aclrtResetDevice(static_cast<int32_t>(ctx->rank)));
}

int main(int argc, char **argv)
{
    Options options;
    if (!ParseOptions(argc, argv, &options)) {
        PrintUsage(argv[0]);
        return 2;
    }
    const std::string route = std::to_string(options.routeIndex);
    if (setenv("A5_CCU_ROUTE_INDEX", route.c_str(), 1) != 0) {
        std::cerr << "Failed to set A5_CCU_ROUTE_INDEX" << std::endl;
        return 2;
    }
    if (setenv("A5_CCU_REMOTE_ONLY", options.remoteOnly ? "1" : "0", 1) != 0 ||
        setenv("A5_CCU_CHANNEL_ONLY", options.channelOnly ? "1" : "0", 1) != 0) {
        std::cerr << "Failed to set probe mode environment" << std::endl;
        return 2;
    }

    aclError aclStatus = aclInit(nullptr);
    if (aclStatus != ACL_SUCCESS) {
        std::cerr << "aclInit failed, status=" << aclStatus << std::endl;
        return 1;
    }
    uint32_t rankSize = 0;
    if (aclrtGetDeviceCount(&rankSize) != ACL_SUCCESS || rankSize != 2) {
        std::cerr << "Exactly two visible devices are required. Set ASCEND_RT_VISIBLE_DEVICES=src,dst; found "
                  << rankSize << std::endl;
        aclFinalize();
        return 2;
    }

    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        aclFinalize();
        return 1;
    }
    void *rootInfoBuffer = nullptr;
    if (aclrtMallocHost(&rootInfoBuffer, sizeof(HcclRootInfo)) != ACL_SUCCESS) {
        aclFinalize();
        return 1;
    }
    auto *rootInfo = static_cast<HcclRootInfo *>(rootInfoBuffer);
    if (HcclGetRootInfo(rootInfo) != HCCL_SUCCESS) {
        aclrtFreeHost(rootInfoBuffer);
        aclFinalize();
        return 1;
    }

    std::atomic<int> failed{0};
    std::vector<ThreadContext> contexts(rankSize);
    std::vector<std::thread> threads;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        contexts[rank] = ThreadContext{rootInfo, rank, rankSize, &options, &failed};
        threads.emplace_back(RunRank, &contexts[rank]);
    }
    for (auto &thread : threads) {
        thread.join();
    }

    if (failed.load() != 0) {
        std::cerr << "Probe failed; skip ACL finalization because a rank may still own partially "
                     "initialized HCCL resources." << std::endl;
        std::cout.flush();
        std::cerr.flush();
        std::_Exit(failed.load());
    }
    aclrtFreeHost(rootInfoBuffer);
    aclFinalize();
    return failed.load();
}
