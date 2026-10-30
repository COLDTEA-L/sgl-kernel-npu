#ifndef A5_CCU_CREATE_TRACE_H
#define A5_CCU_CREATE_TRACE_H
#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <utility>

namespace a5ccu {
// Startup-only diagnostics: preserve the exact API status, including raw CCU
// driver errors. Do not retry a failed resource/registration transaction.
class CreateTrace {
public:
    explicit CreateTrace(std::string context) : context_(std::move(context)) {}
    template<class F> auto Call(const char* stage, F call) const -> decltype(call()) {
        std::fprintf(stderr, "CCU_CREATE_STAGE_BEGIN %s stage=%s\n", context_.c_str(), stage);
        std::fflush(stderr);
        const auto start = std::chrono::steady_clock::now();
        try {
            const auto status = call();
            const auto us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count();
            std::fprintf(stderr, "CCU_CREATE_STAGE_%s %s stage=%s status=%d elapsed_us=%.3f\n",
                static_cast<int>(status) == 0 ? "OK" : "FAILED", context_.c_str(), stage,
                static_cast<int>(status), us);
            std::fflush(stderr);
            return status;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "CCU_CREATE_STAGE_EXCEPTION %s stage=%s error=%s\n",
                context_.c_str(), stage, error.what());
            std::fflush(stderr);
            throw;
        } catch (...) {
            std::fprintf(stderr, "CCU_CREATE_STAGE_EXCEPTION %s stage=%s error=unknown\n",
                context_.c_str(), stage);
            std::fflush(stderr);
            throw;
        }
    }
private:
    std::string context_;
};
}
#endif
