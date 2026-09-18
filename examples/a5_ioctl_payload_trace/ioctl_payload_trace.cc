#include <dlfcn.h>
#include <stdarg.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace {
constexpr size_t MAX_SNAPSHOT = 512;
std::mutex g_mutex;
std::string g_label;
std::unordered_map<unsigned long, uint64_t> g_occurrences;
uint64_t g_events = 0;
thread_local bool g_inside = false;

uint64_t EnvU64(const char *name, uint64_t fallback)
{
    const char *text = std::getenv(name);
    if (text == nullptr || *text == '\0') return fallback;
    char *end = nullptr;
    const auto value = std::strtoull(text, &end, 0);
    return end != text && *end == '\0' ? value : fallback;
}

std::string Escape(const char *text)
{
    std::string result;
    for (const char *p = text == nullptr ? "" : text; *p != '\0'; ++p) {
        if (*p == '\\' || *p == '"') result.push_back('\\');
        if (*p == '\n') result += "\\n";
        else result.push_back(*p);
    }
    return result;
}

std::string Label()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_label;
}

const std::unordered_set<unsigned long> &Requests()
{
    static const auto requests = [] {
        std::unordered_set<unsigned long> result;
        std::istringstream in(std::getenv("A5_IOCTL_PAYLOAD_REQUESTS") == nullptr ?
            "" : std::getenv("A5_IOCTL_PAYLOAD_REQUESTS"));
        std::string token;
        while (std::getline(in, token, ',')) {
            char *end = nullptr;
            const auto request = std::strtoul(token.c_str(), &end, 0);
            if (end != token.c_str() && *end == '\0') result.insert(request);
        }
        return result;
    }();
    return requests;
}

std::string Snapshot(uintptr_t address, size_t length)
{
    length = std::min(length, MAX_SNAPSHOT);
    if (address == 0 || length == 0) return {};
    std::string bytes(length, '\0');
    iovec local{&bytes[0], length};
    iovec remote{reinterpret_cast<void *>(address), length};
    const ssize_t count = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    if (count <= 0) return {};
    static const char digits[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(static_cast<size_t>(count) * 2);
    for (ssize_t i = 0; i < count; ++i) {
        const auto value = static_cast<unsigned char>(bytes[static_cast<size_t>(i)]);
        hex.push_back(digits[value >> 4]);
        hex.push_back(digits[value & 15]);
    }
    return hex;
}

std::string NestedSnapshots(const std::string &payload_hex)
{
    const size_t byte_count = payload_hex.size() / 2;
    if (byte_count < sizeof(uintptr_t)) return "[]";
    std::string bytes(byte_count, '\0');
    for (size_t i = 0; i < byte_count; ++i) {
        bytes[i] = static_cast<char>(std::strtoul(payload_hex.substr(i * 2, 2).c_str(), nullptr, 16));
    }
    std::ostringstream out;
    out << '[';
    size_t emitted = 0;
    const size_t nested_size = std::min(static_cast<size_t>(
        EnvU64("A5_IOCTL_PAYLOAD_NESTED_BYTES", 128)), MAX_SNAPSHOT);
    for (size_t offset = 0; offset + sizeof(uintptr_t) <= bytes.size() && emitted < 8;
         offset += sizeof(uintptr_t)) {
        uintptr_t address = 0;
        std::memcpy(&address, bytes.data() + offset, sizeof(address));
        if (address < 0x10000) continue;
        const std::string nested = Snapshot(address, nested_size);
        if (nested.empty()) continue;
        if (emitted++ != 0) out << ',';
        out << "{\"word_offset\":" << offset << ",\"address\":" << address
            << ",\"captured\":" << nested.size() / 2
            << ",\"hex\":\"" << nested << "\"}";
    }
    out << ']';
    return out.str();
}

std::string Caller(uintptr_t address_value)
{
    Dl_info info{};
    void *address = reinterpret_cast<void *>(address_value);
    if (dladdr(address, &info) == 0) return "{}";
    std::ostringstream out;
    out << "{\"object\":\"" << Escape(info.dli_fname)
        << "\",\"symbol\":\"" << Escape(info.dli_sname)
        << "\",\"address\":" << reinterpret_cast<uintptr_t>(address) << '}';
    return out.str();
}

void Emit(const char *phase, int fd, unsigned long request, uintptr_t argument,
          int status, int saved_errno, uint64_t occurrence, const std::string &label,
          uintptr_t caller)
{
    char link_name[64] = {};
    char fd_path[256] = {};
    std::snprintf(link_name, sizeof(link_name), "/proc/self/fd/%d", fd);
    const ssize_t path_size = readlink(link_name, fd_path, sizeof(fd_path) - 1);
    if (path_size > 0) fd_path[path_size] = '\0';
    const size_t capture = std::min(static_cast<size_t>(_IOC_SIZE(request)),
        static_cast<size_t>(EnvU64("A5_IOCTL_PAYLOAD_CAPTURE_BYTES", MAX_SNAPSHOT)));
    const std::string payload = Snapshot(argument, capture);
    const bool include_caller = occurrence == 1 && std::strcmp(phase, "before") == 0;
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    std::ostringstream out;
    out << "{\"event\":\"IOCTL_PAYLOAD\",\"phase\":\"" << phase
        << "\",\"pid\":" << getpid() << ",\"trace_tid\":" << syscall(SYS_gettid)
        << ",\"trace_ts_ns\":"
        << std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()
        << ",\"trace_label\":\"" << Escape(label.c_str())
        << "\",\"fd\":" << fd << ",\"fd_path\":\""
        << Escape(path_size > 0 ? fd_path : "") << "\",\"request\":\"0x"
        << std::hex << request << std::dec << "\",\"ioc_dir\":" << _IOC_DIR(request)
        << ",\"ioc_type\":" << _IOC_TYPE(request) << ",\"ioc_nr\":" << _IOC_NR(request)
        << ",\"ioc_size\":" << _IOC_SIZE(request) << ",\"argument\":" << argument
        << ",\"occurrence\":" << occurrence << ",\"status\":" << status
        << ",\"errno\":" << saved_errno << ",\"payload_captured\":"
        << payload.size() / 2 << ",\"payload_hex\":\"" << payload
        << "\",\"nested_snapshots\":" << NestedSnapshots(payload)
        << ",\"caller_frames\":[" << (include_caller ? Caller(caller) : "") << "]}";
    const char *prefix = std::getenv("A5_IOCTL_PAYLOAD_TRACE_PREFIX");
    if (prefix == nullptr || *prefix == '\0') return;
    const std::string path = std::string(prefix) + ".pid" + std::to_string(getpid()) + ".jsonl";
    FILE *file = std::fopen(path.c_str(), "a");
    if (file != nullptr) {
        std::fprintf(file, "%s\n", out.str().c_str());
        std::fclose(file);
    }
}

bool Reserve(unsigned long request, uint64_t *occurrence)
{
    const std::string label = Label();
    if (label.empty() || _IOC_SIZE(request) == 0) return false;
    const auto &requests = Requests();
    if (!requests.empty() && requests.count(request) == 0) return false;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_events >= EnvU64("A5_IOCTL_PAYLOAD_MAX_EVENTS", 2048) ||
        g_occurrences[request] >= EnvU64("A5_IOCTL_PAYLOAD_MAX_PER_REQUEST", 64)) return false;
    *occurrence = ++g_occurrences[request];
    ++g_events;
    return true;
}
}  // namespace

extern "C" void A5UrmaTpTraceSetLabel(const char *label)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_label = label == nullptr ? "" : label;
}

extern "C" int ioctl(int fd, unsigned long request, ...)
{
    using Fn = int (*)(int, unsigned long, ...);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "ioctl"));
    if (real == nullptr) { errno = ENOSYS; return -1; }
    va_list args;
    va_start(args, request);
    const uintptr_t argument = va_arg(args, uintptr_t);
    va_end(args);
    uint64_t occurrence = 0;
    const bool trace = !g_inside && Reserve(request, &occurrence);
    const std::string label = trace ? Label() : "";
    const uintptr_t caller = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
    if (trace) { g_inside = true; Emit("before", fd, request, argument, 0, 0, occurrence, label, caller); g_inside = false; }
    const int status = real(fd, request, argument);
    const int saved_errno = errno;
    if (trace) { g_inside = true; Emit("after", fd, request, argument, status, saved_errno, occurrence, label, caller); g_inside = false; }
    errno = saved_errno;
    return status;
}
