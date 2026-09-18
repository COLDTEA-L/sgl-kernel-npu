#include <dlfcn.h>
#include <execinfo.h>
#include <stdarg.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <urma_api.h>
#include <urma_cmd.h>

namespace {
std::mutex g_traceMutex;
std::mutex g_labelMutex;
std::string g_traceLabel;
thread_local bool g_insideTrace = false;
thread_local uint32_t g_publicGetTpListDepth = 0;
constexpr size_t MAX_PRIVATE_SNAPSHOT = 4096;
constexpr size_t MAX_IOCTL_SNAPSHOT = 512;
std::mutex g_ioctlMutex;
std::unordered_map<unsigned long, uint64_t> g_ioctlOccurrences;
uint64_t g_ioctlEvents = 0;

template <typename T>
std::string Hex(const T &value)
{
    const auto *bytes = reinterpret_cast<const unsigned char *>(&value);
    static const char digits[] = "0123456789abcdef";
    std::string result(sizeof(T) * 2, '0');
    for (size_t i = 0; i < sizeof(T); ++i) {
        result[i * 2] = digits[bytes[i] >> 4];
        result[i * 2 + 1] = digits[bytes[i] & 0xf];
    }
    return result;
}

void Emit(const std::string &line)
{
    std::string enriched = line;
    if (!enriched.empty() && enriched.back() == '}') {
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
        std::ostringstream metadata;
        metadata << ",\"trace_ts_ns\":" << ns
                 << ",\"trace_tid\":" << static_cast<long>(syscall(SYS_gettid));
        enriched.insert(enriched.size() - 1, metadata.str());
    }
    std::lock_guard<std::mutex> guard(g_traceMutex);
    const char *prefix = std::getenv("A5_URMA_TP_TRACE_PREFIX");
    if (prefix == nullptr || prefix[0] == '\0') {
        std::fprintf(stderr, "%s\n", enriched.c_str());
        std::fflush(stderr);
        return;
    }
    const std::string path = std::string(prefix) + ".pid" + std::to_string(getpid()) + ".jsonl";
    FILE *file = std::fopen(path.c_str(), "a");
    if (file != nullptr) {
        std::fprintf(file, "%s\n", enriched.c_str());
        std::fclose(file);
    }
}

std::string CurrentTraceLabel()
{
    std::lock_guard<std::mutex> guard(g_labelMutex);
    return g_traceLabel;
}

std::string JsonEscape(const char *value);
std::string CallerFrames();
void AppendTraceLabel(std::ostringstream &out);

struct PrivateBufferSnapshot {
    uint64_t address = 0;
    uint32_t requested = 0;
    size_t captured = 0;
    long readStatus = 0;
    std::string hex;
};

void AppendPrivateBuffer(std::ostringstream &out, const char *name,
                         const PrivateBufferSnapshot &snapshot);

PrivateBufferSnapshot SnapshotBuffer(uint64_t address, uint32_t length)
{
    PrivateBufferSnapshot snapshot;
    snapshot.address = address;
    snapshot.requested = length;
    if (address == 0 || length == 0) return snapshot;
    const size_t capture = std::min(static_cast<size_t>(length), MAX_PRIVATE_SNAPSHOT);
    std::string bytes(capture, '\0');
    iovec local{&bytes[0], capture};
    iovec remote{reinterpret_cast<void *>(static_cast<uintptr_t>(address)), capture};
    snapshot.readStatus = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    if (snapshot.readStatus <= 0) return snapshot;
    snapshot.captured = static_cast<size_t>(snapshot.readStatus);
    static const char digits[] = "0123456789abcdef";
    snapshot.hex.reserve(snapshot.captured * 2);
    for (size_t i = 0; i < snapshot.captured; ++i) {
        const auto value = static_cast<unsigned char>(bytes[i]);
        snapshot.hex.push_back(digits[value >> 4]);
        snapshot.hex.push_back(digits[value & 0xf]);
    }
    return snapshot;
}

uint64_t ParsePositiveEnv(const char *name, uint64_t fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 0);
    return end != value && *end == '\0' && parsed > 0 ? parsed : fallback;
}

const std::unordered_set<unsigned long> &IoctlRequestFilter()
{
    static const std::unordered_set<unsigned long> requests = [] {
        std::unordered_set<unsigned long> result;
        const char *text = std::getenv("A5_IOCTL_PAYLOAD_REQUESTS");
        if (text == nullptr) return result;
        std::istringstream input(text);
        std::string token;
        while (std::getline(input, token, ',')) {
            char *end = nullptr;
            const unsigned long request = std::strtoul(token.c_str(), &end, 0);
            if (end != token.c_str() && *end == '\0') result.insert(request);
        }
        return result;
    }();
    return requests;
}

bool ShouldTraceIoctl(unsigned long request)
{
    if (CurrentTraceLabel().empty()) return false;
    const auto &filter = IoctlRequestFilter();
    if (!filter.empty() && filter.count(request) == 0) return false;
    if (_IOC_SIZE(request) == 0) return false;
    std::lock_guard<std::mutex> guard(g_ioctlMutex);
    return g_ioctlEvents < ParsePositiveEnv("A5_IOCTL_PAYLOAD_MAX_EVENTS", 1024);
}

void EmitIoctlPayload(const char *phase, int fd, unsigned long request, uintptr_t argument,
                      int status, int savedErrno, uint64_t occurrence)
{
    const size_t configured = static_cast<size_t>(
        ParsePositiveEnv("A5_IOCTL_PAYLOAD_CAPTURE_BYTES", MAX_IOCTL_SNAPSHOT));
    const uint32_t capture = static_cast<uint32_t>(
        std::min({static_cast<size_t>(_IOC_SIZE(request)), configured, MAX_IOCTL_SNAPSHOT}));
    const PrivateBufferSnapshot snapshot = SnapshotBuffer(argument, capture);
    char fdPath[256] = {};
    char linkPath[64] = {};
    std::snprintf(linkPath, sizeof(linkPath), "/proc/self/fd/%d", fd);
    const ssize_t pathLength = readlink(linkPath, fdPath, sizeof(fdPath) - 1);
    if (pathLength > 0) fdPath[pathLength] = '\0';
    std::ostringstream out;
    out << "{\"event\":\"IOCTL_PAYLOAD\",\"phase\":\"" << phase
        << "\",\"pid\":" << getpid() << ",\"fd\":" << fd
        << ",\"fd_path\":\"" << JsonEscape(pathLength > 0 ? fdPath : "")
        << "\",\"request\":\"0x" << std::hex << request << std::dec
        << "\",\"ioc_dir\":" << _IOC_DIR(request)
        << ",\"ioc_type\":" << _IOC_TYPE(request)
        << ",\"ioc_nr\":" << _IOC_NR(request)
        << ",\"ioc_size\":" << _IOC_SIZE(request)
        << ",\"argument\":" << argument << ",\"occurrence\":" << occurrence
        << ",\"status\":" << status << ",\"errno\":" << savedErrno;
    AppendPrivateBuffer(out, "payload", snapshot);
    out << ",\"caller_frames\":" << CallerFrames();
    AppendTraceLabel(out);
    out << '}';
    Emit(out.str());
}

struct UdataSnapshot {
    bool present = false;
    PrivateBufferSnapshot input;
    PrivateBufferSnapshot output;
};

UdataSnapshot SnapshotUdata(const urma_cmd_udrv_priv_t *udata)
{
    UdataSnapshot snapshot;
    if (udata == nullptr) return snapshot;
    snapshot.present = true;
    snapshot.input = SnapshotBuffer(udata->in_addr, udata->in_len);
    snapshot.output = SnapshotBuffer(udata->out_addr, udata->out_len);
    return snapshot;
}

void AppendPrivateBuffer(std::ostringstream &out, const char *name,
                         const PrivateBufferSnapshot &snapshot)
{
    out << ",\"" << name << "_addr\":" << snapshot.address
        << ",\"" << name << "_len\":" << snapshot.requested
        << ",\"" << name << "_captured\":" << snapshot.captured
        << ",\"" << name << "_read_status\":" << snapshot.readStatus
        << ",\"" << name << "_hex\":\"" << snapshot.hex << '"';
}

void EmitPrivateTpRequest(const char *api, const char *phase, int status,
                          const urma_get_tp_cfg_t *cfg, const UdataSnapshot &udata,
                          const std::string &commandRaw)
{
    std::ostringstream out;
    out << "{\"event\":\"PRIVATE_TP_REQUEST\",\"api\":\"" << api
        << "\",\"phase\":\"" << phase << "\",\"pid\":" << getpid()
        << ",\"status\":" << status << ",\"udata_present\":"
        << (udata.present ? "true" : "false");
    if (cfg != nullptr) {
        out << ",\"flag\":" << cfg->flag.value
            << ",\"trans_mode\":" << static_cast<uint32_t>(cfg->trans_mode)
            << ",\"local_eid_raw\":\"" << Hex(cfg->local_eid)
            << "\",\"peer_eid_raw\":\"" << Hex(cfg->peer_eid) << '"';
    }
    AppendPrivateBuffer(out, "udata_in", udata.input);
    AppendPrivateBuffer(out, "udata_out", udata.output);
    out << ",\"command_raw\":\"" << commandRaw << '"'
        << ",\"caller_frames\":" << CallerFrames();
    AppendTraceLabel(out);
    out << '}';
    Emit(out.str());
}

void AppendTraceLabel(std::ostringstream &out)
{
    const std::string label = CurrentTraceLabel();
    if (!label.empty()) out << ",\"trace_label\":\"" << JsonEscape(label.c_str()) << "\"";
}

std::string Handles(const urma_tp_info_t *list, uint32_t count)
{
    std::ostringstream out;
    out << '[';
    for (uint32_t i = 0; list != nullptr && i < count; ++i) {
        if (i != 0) out << ',';
        out << list[i].tp_handle;
    }
    out << ']';
    return out.str();
}

std::string JsonEscape(const char *value)
{
    std::ostringstream out;
    for (const char *cursor = value == nullptr ? "" : value; *cursor != '\0'; ++cursor) {
        if (*cursor == '\\' || *cursor == '"') out << '\\';
        out << *cursor;
    }
    return out.str();
}

std::string CallerFrames()
{
    void *frames[20] = {};
    const int count = backtrace(frames, static_cast<int>(sizeof(frames) / sizeof(frames[0])));
    std::ostringstream out;
    out << '[';
    for (int i = 2; i < count; ++i) {
        Dl_info info = {};
        if (dladdr(frames[i], &info) == 0 || info.dli_fname == nullptr) continue;
        if (out.tellp() > 1) out << ',';
        const auto address = reinterpret_cast<uintptr_t>(frames[i]);
        const auto base = reinterpret_cast<uintptr_t>(info.dli_fbase);
        out << "{\"object\":\"" << JsonEscape(info.dli_fname)
            << "\",\"object_offset\":\"0x" << std::hex << (address - base) << std::dec << "\"";
        if (info.dli_sname != nullptr && info.dli_saddr != nullptr) {
            const auto symbol = reinterpret_cast<uintptr_t>(info.dli_saddr);
            out << ",\"symbol\":\"" << JsonEscape(info.dli_sname)
                << "\",\"symbol_offset\":\"0x" << std::hex << (address - symbol)
                << std::dec << "\"";
        }
        out << '}';
    }
    out << ']';
    return out.str();
}

void EmitTpList(const char *api, const urma_get_tp_cfg_t *cfg, uint32_t requested,
                int status, uint32_t returned, const urma_tp_info_t *list, const void *ctx)
{
    if (cfg == nullptr) return;
    std::ostringstream out;
    out << "{\"event\":\"GET_TP_LIST\",\"api\":\"" << api
        << "\",\"pid\":" << getpid() << ",\"status\":" << status
        << ",\"context\":\"" << ctx << "\""
        << ",\"flag\":" << cfg->flag.value
        << ",\"trans_mode\":" << static_cast<uint32_t>(cfg->trans_mode)
        << ",\"requested\":" << requested << ",\"returned\":" << returned
        << ",\"local_eid_raw\":\"" << Hex(cfg->local_eid)
        << "\",\"peer_eid_raw\":\"" << Hex(cfg->peer_eid)
        << "\",\"tp_handles\":" << Handles(list, returned)
        << ",\"caller_frames\":" << CallerFrames();
    AppendTraceLabel(out);
    out << '}';
    Emit(out.str());
}

void EmitTpAttr(const char *api, uint64_t handle, int status, uint8_t count,
                uint32_t bitmap, const urma_tp_attr_value_t *attrs)
{
    std::ostringstream out;
    out << "{\"event\":\"GET_TP_ATTR\",\"api\":\"" << api
        << "\",\"pid\":" << getpid() << ",\"status\":" << status
        << ",\"tp_handle\":" << handle << ",\"attr_count\":"
        << static_cast<uint32_t>(count) << ",\"attr_bitmap\":" << bitmap;
    if (attrs != nullptr && count != 0) {
        out << ",\"retry_times_init\":" << static_cast<uint32_t>(attrs->retry_times_init)
            << ",\"address_type\":" << static_cast<uint32_t>(attrs->at)
            << ",\"sip_raw\":\"" << Hex(attrs->sip)
            << "\",\"dip_raw\":\"" << Hex(attrs->dip)
            << "\",\"sma_raw\":\"" << Hex(attrs->sma)
            << "\",\"dma_raw\":\"" << Hex(attrs->dma)
            << "\",\"vlan_id\":" << attrs->vlan_id
            << ",\"vlan_en\":" << static_cast<uint32_t>(attrs->vlan_en)
            << ",\"dscp\":" << static_cast<uint32_t>(attrs->dscp)
            << ",\"sl\":" << static_cast<uint32_t>(attrs->sl)
            << ",\"ttl\":" << static_cast<uint32_t>(attrs->ttl)
            << ",\"attr_raw\":\"";
        const auto *bytes = reinterpret_cast<const unsigned char *>(attrs);
        static const char digits[] = "0123456789abcdef";
        for (size_t i = 0; i < static_cast<size_t>(count) * sizeof(*attrs); ++i) {
            out << digits[bytes[i] >> 4] << digits[bytes[i] & 0xf];
        }
        out << '"';
    }
    AppendTraceLabel(out);
    out << '}';
    Emit(out.str());
}

void ProbeReturnedTpAttrs(const urma_context_t *ctx, const urma_tp_info_t *list,
                          uint32_t count, const char *sourceApi)
{
    using Fn = urma_status_t (*)(const urma_context_t *, uint64_t, uint8_t *, uint32_t *,
                                 urma_tp_attr_value_t *);
    static Fn realGetAttr = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_get_tp_attr"));
    if (realGetAttr == nullptr || ctx == nullptr || list == nullptr) return;
    for (uint32_t i = 0; i < count; ++i) {
        uint8_t attrCount = 0;
        uint32_t attrBitmap = 0;
        urma_tp_attr_value_t attr = {};
        const urma_status_t status = realGetAttr(ctx, list[i].tp_handle, &attrCount,
                                                 &attrBitmap, &attr);
        const std::string api = std::string("trace_probe_after_") + sourceApi;
        EmitTpAttr(api.c_str(), list[i].tp_handle, static_cast<int>(status), attrCount,
                   attrBitmap, &attr);
    }
}

void EmitSetTpAttr(const char *api, uint64_t handle, int status, uint8_t count,
                   uint32_t bitmap, const urma_tp_attr_value_t *attrs)
{
    std::ostringstream out;
    out << "{\"event\":\"SET_TP_ATTR\",\"api\":\"" << api
        << "\",\"pid\":" << getpid() << ",\"status\":" << status
        << ",\"tp_handle\":" << handle << ",\"attr_count\":"
        << static_cast<uint32_t>(count) << ",\"attr_bitmap\":" << bitmap;
    if (attrs != nullptr) out << ",\"attr_raw\":\"" << Hex(*attrs) << '"';
    AppendTraceLabel(out);
    out << '}';
    Emit(out.str());
}

void EmitModifyTp(const char *api, uint32_t tpn, int status, const urma_tp_cfg_t *cfg,
                  const urma_tp_attr_t *attr, urma_tp_attr_mask_t mask)
{
    std::ostringstream out;
    out << "{\"event\":\"MODIFY_TP\",\"api\":\"" << api
        << "\",\"pid\":" << getpid() << ",\"status\":" << status
        << ",\"tpn\":" << tpn << ",\"mask\":" << mask.value;
    if (cfg != nullptr) {
        out << ",\"cfg_flag\":" << cfg->flag.value
            << ",\"trans_mode\":" << static_cast<uint32_t>(cfg->trans_mode);
    }
    if (attr != nullptr) {
        out << ",\"mod_flag\":" << attr->flag.value
            << ",\"spray_en\":" << attr->flag.bs.spray_en
            << ",\"peer_tpn\":" << attr->peer_tpn
            << ",\"state\":" << static_cast<uint32_t>(attr->state)
            << ",\"local_net_addr_idx\":" << attr->local_net_addr_idx
            << ",\"data_udp_start\":" << attr->data_udp_start
            << ",\"ack_udp_start\":" << attr->ack_udp_start
            << ",\"udp_range\":" << static_cast<uint32_t>(attr->udp_range)
            << ",\"hop_limit\":" << static_cast<uint32_t>(attr->hop_limit)
            << ",\"flow_label\":" << attr->flow_label
            << ",\"port_id\":" << static_cast<uint32_t>(attr->port_id)
            << ",\"attr_raw\":\"" << Hex(*attr) << '"';
    }
    AppendTraceLabel(out);
    out << '}';
    Emit(out.str());
}

void EmitExchange(const urma_get_tp_cfg_t *cfg, uint64_t localHandle, uint32_t txPsn,
                  int status, uint64_t peerHandle, uint32_t rxPsn)
{
    std::ostringstream out;
    out << "{\"event\":\"EXCHANGE_TP_INFO\",\"api\":\"urma_cmd_exchange_tp_info\""
        << ",\"pid\":" << getpid() << ",\"status\":" << status
        << ",\"local_tp_handle\":" << localHandle << ",\"peer_tp_handle\":" << peerHandle
        << ",\"tx_psn\":" << txPsn << ",\"rx_psn\":" << rxPsn;
    if (cfg != nullptr) {
        out << ",\"flag\":" << cfg->flag.value
            << ",\"trans_mode\":" << static_cast<uint32_t>(cfg->trans_mode)
            << ",\"local_eid_raw\":\"" << Hex(cfg->local_eid)
            << "\",\"peer_eid_raw\":\"" << Hex(cfg->peer_eid) << '"';
    }
    AppendTraceLabel(out);
    out << '}';
    Emit(out.str());
}

void EmitTpActivation(const char *op, const void *ctx, const urma_eid_t *remoteEid,
                      uint32_t remoteUasid, uint32_t remoteId, uint32_t transMode,
                      uint32_t tpType, const urma_active_tp_cfg_t *cfg,
                      const urma_target_jetty_t *target, int status)
{
    std::ostringstream out;
    out << "{\"event\":\"TP_ACTIVATION\",\"op\":\"" << op
        << "\",\"pid\":" << getpid() << ",\"context\":\"" << ctx << "\""
        << ",\"status\":" << status << ",\"remote_uasid\":" << remoteUasid
        << ",\"remote_id\":" << remoteId << ",\"trans_mode\":" << transMode
        << ",\"tp_type\":" << tpType;
    if (remoteEid != nullptr) out << ",\"remote_eid_raw\":\"" << Hex(*remoteEid) << "\"";
    if (cfg != nullptr) {
        out << ",\"active_tp_handle\":" << cfg->tp_handle
            << ",\"active_peer_tp_handle\":" << cfg->peer_tp_handle
            << ",\"active_tag\":" << cfg->tag
            << ",\"active_tx_psn\":" << cfg->tp_attr.tx_psn
            << ",\"active_rx_psn\":" << cfg->tp_attr.rx_psn;
    }
    if (target != nullptr) {
        out << ",\"target_handle\":" << target->handle
            << ",\"target_tpn\":" << target->tp.tpn
            << ",\"target_id\":" << target->id.id
            << ",\"target_uasid\":" << target->id.uasid
            << ",\"target_eid_raw\":\"" << Hex(target->id.eid) << "\"";
    }
    out << ",\"caller_frames\":" << CallerFrames();
    AppendTraceLabel(out);
    out << '}';
    Emit(out.str());
}

void EmitJettyBind(const char *op, const urma_jetty_t *jetty,
                   const urma_target_jetty_t *target, const urma_active_tp_cfg_t *cfg,
                   int status)
{
    std::ostringstream out;
    out << "{\"event\":\"TP_ACTIVATION\",\"op\":\"" << op
        << "\",\"pid\":" << getpid() << ",\"status\":" << status;
    if (jetty != nullptr) {
        out << ",\"local_jetty_handle\":" << jetty->handle
            << ",\"local_jetty_id\":" << jetty->jetty_id.id
            << ",\"local_eid_raw\":\"" << Hex(jetty->jetty_id.eid) << "\"";
    }
    if (target != nullptr) {
        out << ",\"target_handle\":" << target->handle
            << ",\"target_tpn\":" << target->tp.tpn
            << ",\"target_id\":" << target->id.id
            << ",\"target_eid_raw\":\"" << Hex(target->id.eid) << "\"";
    }
    if (cfg != nullptr) {
        out << ",\"active_tp_handle\":" << cfg->tp_handle
            << ",\"active_peer_tp_handle\":" << cfg->peer_tp_handle
            << ",\"active_tag\":" << cfg->tag
            << ",\"active_tx_psn\":" << cfg->tp_attr.tx_psn
            << ",\"active_rx_psn\":" << cfg->tp_attr.rx_psn;
    }
    out << ",\"caller_frames\":" << CallerFrames();
    AppendTraceLabel(out);
    out << '}';
    Emit(out.str());
}
} // namespace

extern "C" void A5UrmaTpTraceSetLabel(const char *label)
{
    {
        std::lock_guard<std::mutex> guard(g_labelMutex);
        g_traceLabel = label == nullptr ? "" : label;
    }
    std::ostringstream out;
    out << "{\"event\":\"TRACE_SCOPE\",\"pid\":" << getpid()
        << ",\"trace_label\":\"" << JsonEscape(label == nullptr ? "" : label) << "\"}";
    Emit(out.str());
}

extern "C" int ioctl(int fd, unsigned long request, ...)
{
    using Fn = int (*)(int, unsigned long, ...);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "ioctl"));
    if (real == nullptr) {
        errno = ENOSYS;
        return -1;
    }
    va_list args;
    va_start(args, request);
    const uintptr_t argument = va_arg(args, uintptr_t);
    va_end(args);

    const bool trace = !g_insideTrace && ShouldTraceIoctl(request);
    uint64_t occurrence = 0;
    if (trace) {
        std::lock_guard<std::mutex> guard(g_ioctlMutex);
        occurrence = ++g_ioctlOccurrences[request];
        ++g_ioctlEvents;
        g_insideTrace = true;
        EmitIoctlPayload("before", fd, request, argument, 0, 0, occurrence);
        g_insideTrace = false;
    }
    const int status = real(fd, request, argument);
    const int savedErrno = errno;
    if (trace) {
        g_insideTrace = true;
        EmitIoctlPayload("after", fd, request, argument, status, savedErrno, occurrence);
        g_insideTrace = false;
    }
    errno = savedErrno;
    return status;
}

extern "C" urma_status_t urma_get_tp_list(urma_context_t *ctx, urma_get_tp_cfg_t *cfg,
    uint32_t *tp_cnt, urma_tp_info_t *tp_list)
{
    using Fn = urma_status_t (*)(urma_context_t *, urma_get_tp_cfg_t *, uint32_t *, urma_tp_info_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_get_tp_list"));
    if (real == nullptr) return static_cast<urma_status_t>(-1);
    const uint32_t requested = tp_cnt == nullptr ? 0 : *tp_cnt;
    ++g_publicGetTpListDepth;
    const urma_status_t status = real(ctx, cfg, tp_cnt, tp_list);
    --g_publicGetTpListDepth;
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitTpList("urma_get_tp_list", cfg, requested, static_cast<int>(status),
                   tp_cnt == nullptr ? 0 : *tp_cnt, tp_list, ctx);
        if (status == URMA_SUCCESS) {
            ProbeReturnedTpAttrs(ctx, tp_list, tp_cnt == nullptr ? 0 : *tp_cnt,
                                 "urma_get_tp_list");
        }
        g_insideTrace = false;
    }
    return status;
}

extern "C" int urma_cmd_get_tp_list(urma_context_t *ctx, urma_get_tp_cfg_t *cfg,
    uint32_t *tp_cnt, urma_tp_info_t *tp_list, urma_cmd_udrv_priv_t *udata)
{
    using Fn = int (*)(urma_context_t *, urma_get_tp_cfg_t *, uint32_t *, urma_tp_info_t *,
                       urma_cmd_udrv_priv_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_cmd_get_tp_list"));
    if (real == nullptr) return -1;
    const uint32_t requested = tp_cnt == nullptr ? 0 : *tp_cnt;
    const UdataSnapshot before = SnapshotUdata(udata);
    const int status = real(ctx, cfg, tp_cnt, tp_list, udata);
    const UdataSnapshot after = SnapshotUdata(udata);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitPrivateTpRequest("urma_cmd_get_tp_list", "before", 0, cfg, before, "");
        EmitPrivateTpRequest("urma_cmd_get_tp_list", "after", status, cfg, after, "");
        EmitTpList("urma_cmd_get_tp_list", cfg, requested, status,
                   tp_cnt == nullptr ? 0 : *tp_cnt, tp_list, ctx);
        if (status == 0 && g_publicGetTpListDepth == 0) {
            ProbeReturnedTpAttrs(ctx, tp_list, tp_cnt == nullptr ? 0 : *tp_cnt,
                                 "urma_cmd_get_tp_list");
        }
        g_insideTrace = false;
    }
    return status;
}

extern "C" int urma_ioctl_get_tp_list(int ioctlFd, urma_cmd_get_tp_list_t *arg)
{
    using Fn = int (*)(int, urma_cmd_get_tp_list_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_ioctl_get_tp_list"));
    if (real == nullptr) return -1;
    const std::string beforeRaw = arg == nullptr ? "" : Hex(*arg);
    const UdataSnapshot before = arg == nullptr ? UdataSnapshot{} : SnapshotUdata(&arg->udata);
    urma_get_tp_cfg_t cfg = {};
    if (arg != nullptr) {
        cfg.flag.value = arg->in.flag;
        cfg.trans_mode = static_cast<urma_transport_mode_t>(arg->in.trans_mode);
        (void)std::memcpy(&cfg.local_eid, arg->in.local_eid, sizeof(cfg.local_eid));
        (void)std::memcpy(&cfg.peer_eid, arg->in.peer_eid, sizeof(cfg.peer_eid));
    }
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitPrivateTpRequest("urma_ioctl_get_tp_list", "before", 0,
                             arg == nullptr ? nullptr : &cfg, before, beforeRaw);
        g_insideTrace = false;
    }
    const int status = real(ioctlFd, arg);
    if (!g_insideTrace) {
        g_insideTrace = true;
        const UdataSnapshot after = arg == nullptr ? UdataSnapshot{} : SnapshotUdata(&arg->udata);
        EmitPrivateTpRequest("urma_ioctl_get_tp_list", "after", status,
                             arg == nullptr ? nullptr : &cfg, after,
                             arg == nullptr ? "" : Hex(*arg));
        g_insideTrace = false;
    }
    return status;
}

extern "C" int udma_u_ctrlq_get_tp_list(urma_context_t *ctx, urma_get_tp_cfg_t *cfg,
    uint32_t *tpCnt, urma_tp_info_t *tpList)
{
    using Fn = int (*)(urma_context_t *, urma_get_tp_cfg_t *, uint32_t *, urma_tp_info_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "udma_u_ctrlq_get_tp_list"));
    if (real == nullptr) return -1;
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitPrivateTpRequest("udma_u_ctrlq_get_tp_list", "before", 0, cfg,
                             UdataSnapshot{}, "");
        g_insideTrace = false;
    }
    const int status = real(ctx, cfg, tpCnt, tpList);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitPrivateTpRequest("udma_u_ctrlq_get_tp_list", "after", status, cfg,
                             UdataSnapshot{}, "");
        g_insideTrace = false;
    }
    return status;
}

extern "C" urma_status_t urma_get_tp_attr(const urma_context_t *ctx, uint64_t handle,
    uint8_t *count, uint32_t *bitmap, urma_tp_attr_value_t *attrs)
{
    using Fn = urma_status_t (*)(const urma_context_t *, uint64_t, uint8_t *, uint32_t *,
                                 urma_tp_attr_value_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_get_tp_attr"));
    if (real == nullptr) return static_cast<urma_status_t>(-1);
    const urma_status_t status = real(ctx, handle, count, bitmap, attrs);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitTpAttr("urma_get_tp_attr", handle, static_cast<int>(status),
                   count == nullptr ? 0 : *count, bitmap == nullptr ? 0 : *bitmap, attrs);
        g_insideTrace = false;
    }
    return status;
}

extern "C" int urma_cmd_get_tp_attr(const urma_context_t *ctx, uint64_t handle,
    uint8_t *count, uint32_t *bitmap, urma_tp_attr_value_t *attrs, urma_cmd_udrv_priv_t *udata)
{
    using Fn = int (*)(const urma_context_t *, uint64_t, uint8_t *, uint32_t *,
                       urma_tp_attr_value_t *, urma_cmd_udrv_priv_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_cmd_get_tp_attr"));
    if (real == nullptr) return -1;
    const int status = real(ctx, handle, count, bitmap, attrs, udata);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitTpAttr("urma_cmd_get_tp_attr", handle, status, count == nullptr ? 0 : *count,
                   bitmap == nullptr ? 0 : *bitmap, attrs);
        g_insideTrace = false;
    }
    return status;
}

extern "C" urma_status_t urma_set_tp_attr(const urma_context_t *ctx, uint64_t handle,
    uint8_t count, uint32_t bitmap, const urma_tp_attr_value_t *attrs)
{
    using Fn = urma_status_t (*)(const urma_context_t *, uint64_t, uint8_t, uint32_t,
                                 const urma_tp_attr_value_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_set_tp_attr"));
    if (real == nullptr) return static_cast<urma_status_t>(-1);
    const urma_status_t status = real(ctx, handle, count, bitmap, attrs);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitSetTpAttr("urma_set_tp_attr", handle, static_cast<int>(status), count, bitmap, attrs);
        g_insideTrace = false;
    }
    return status;
}

extern "C" int urma_cmd_set_tp_attr(const urma_context_t *ctx, uint64_t handle,
    uint8_t count, uint32_t bitmap, const urma_tp_attr_value_t *attrs,
    urma_cmd_udrv_priv_t *udata)
{
    using Fn = int (*)(const urma_context_t *, uint64_t, uint8_t, uint32_t,
                       const urma_tp_attr_value_t *, urma_cmd_udrv_priv_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_cmd_set_tp_attr"));
    if (real == nullptr) return -1;
    const int status = real(ctx, handle, count, bitmap, attrs, udata);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitSetTpAttr("urma_cmd_set_tp_attr", handle, status, count, bitmap, attrs);
        g_insideTrace = false;
    }
    return status;
}

extern "C" int urma_modify_tp(urma_context_t *ctx, uint32_t tpn, urma_tp_cfg_t *cfg,
    urma_tp_attr_t *attr, urma_tp_attr_mask_t mask)
{
    using Fn = int (*)(urma_context_t *, uint32_t, urma_tp_cfg_t *, urma_tp_attr_t *,
                       urma_tp_attr_mask_t);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_modify_tp"));
    if (real == nullptr) return -1;
    const int status = real(ctx, tpn, cfg, attr, mask);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitModifyTp("urma_modify_tp", tpn, status, cfg, attr, mask);
        g_insideTrace = false;
    }
    return status;
}

extern "C" int urma_cmd_modify_tp(urma_context_t *ctx, uint32_t tpn, urma_tp_cfg_t *cfg,
    urma_tp_attr_t *attr, urma_tp_attr_mask_t mask)
{
    using Fn = int (*)(urma_context_t *, uint32_t, urma_tp_cfg_t *, urma_tp_attr_t *,
                       urma_tp_attr_mask_t);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_cmd_modify_tp"));
    if (real == nullptr) return -1;
    const int status = real(ctx, tpn, cfg, attr, mask);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitModifyTp("urma_cmd_modify_tp", tpn, status, cfg, attr, mask);
        g_insideTrace = false;
    }
    return status;
}

extern "C" int urma_cmd_exchange_tp_info(urma_context_t *ctx, urma_get_tp_cfg_t *cfg,
    uint64_t localHandle, uint32_t txPsn, uint64_t *peerHandle, uint32_t *rxPsn)
{
    using Fn = int (*)(urma_context_t *, urma_get_tp_cfg_t *, uint64_t, uint32_t,
                       uint64_t *, uint32_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_cmd_exchange_tp_info"));
    if (real == nullptr) return -1;
    const int status = real(ctx, cfg, localHandle, txPsn, peerHandle, rxPsn);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitExchange(cfg, localHandle, txPsn, status,
                     peerHandle == nullptr ? 0 : *peerHandle, rxPsn == nullptr ? 0 : *rxPsn);
        g_insideTrace = false;
    }
    return status;
}

extern "C" urma_target_jetty_t *urma_import_jfr(urma_context_t *ctx, urma_rjfr_t *rjfr,
    urma_token_t *token)
{
    using Fn = urma_target_jetty_t *(*)(urma_context_t *, urma_rjfr_t *, urma_token_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_import_jfr"));
    if (real == nullptr) return nullptr;
    urma_target_jetty_t *target = real(ctx, rjfr, token);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitTpActivation("urma_import_jfr", ctx, rjfr == nullptr ? nullptr : &rjfr->jfr_id.eid,
                         rjfr == nullptr ? 0 : rjfr->jfr_id.uasid,
                         rjfr == nullptr ? 0 : rjfr->jfr_id.id,
                         rjfr == nullptr ? 0 : static_cast<uint32_t>(rjfr->trans_mode),
                         rjfr == nullptr ? 0 : static_cast<uint32_t>(rjfr->tp_type),
                         nullptr, target, target == nullptr ? -1 : 0);
        g_insideTrace = false;
    }
    return target;
}

extern "C" urma_target_jetty_t *urma_import_jfr_ex(urma_context_t *ctx, urma_rjfr_t *rjfr,
    urma_token_t *token, urma_import_jfr_ex_cfg_t *cfg)
{
    using Fn = urma_target_jetty_t *(*)(urma_context_t *, urma_rjfr_t *, urma_token_t *,
                                        urma_import_jfr_ex_cfg_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_import_jfr_ex"));
    if (real == nullptr) return nullptr;
    urma_target_jetty_t *target = real(ctx, rjfr, token, cfg);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitTpActivation("urma_import_jfr_ex", ctx, rjfr == nullptr ? nullptr : &rjfr->jfr_id.eid,
                         rjfr == nullptr ? 0 : rjfr->jfr_id.uasid,
                         rjfr == nullptr ? 0 : rjfr->jfr_id.id,
                         rjfr == nullptr ? 0 : static_cast<uint32_t>(rjfr->trans_mode),
                         rjfr == nullptr ? 0 : static_cast<uint32_t>(rjfr->tp_type),
                         cfg, target, target == nullptr ? -1 : 0);
        g_insideTrace = false;
    }
    return target;
}

extern "C" urma_target_jetty_t *urma_import_jetty(urma_context_t *ctx, urma_rjetty_t *rjetty,
    urma_token_t *token)
{
    using Fn = urma_target_jetty_t *(*)(urma_context_t *, urma_rjetty_t *, urma_token_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_import_jetty"));
    if (real == nullptr) return nullptr;
    urma_target_jetty_t *target = real(ctx, rjetty, token);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitTpActivation("urma_import_jetty", ctx,
                         rjetty == nullptr ? nullptr : &rjetty->jetty_id.eid,
                         rjetty == nullptr ? 0 : rjetty->jetty_id.uasid,
                         rjetty == nullptr ? 0 : rjetty->jetty_id.id,
                         rjetty == nullptr ? 0 : static_cast<uint32_t>(rjetty->trans_mode),
                         rjetty == nullptr ? 0 : static_cast<uint32_t>(rjetty->tp_type),
                         nullptr, target, target == nullptr ? -1 : 0);
        g_insideTrace = false;
    }
    return target;
}

extern "C" urma_target_jetty_t *urma_import_jetty_ex(urma_context_t *ctx,
    urma_rjetty_t *rjetty, urma_token_t *token, urma_import_jetty_ex_cfg_t *cfg)
{
    using Fn = urma_target_jetty_t *(*)(urma_context_t *, urma_rjetty_t *, urma_token_t *,
                                        urma_import_jetty_ex_cfg_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_import_jetty_ex"));
    if (real == nullptr) return nullptr;
    urma_target_jetty_t *target = real(ctx, rjetty, token, cfg);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitTpActivation("urma_import_jetty_ex", ctx,
                         rjetty == nullptr ? nullptr : &rjetty->jetty_id.eid,
                         rjetty == nullptr ? 0 : rjetty->jetty_id.uasid,
                         rjetty == nullptr ? 0 : rjetty->jetty_id.id,
                         rjetty == nullptr ? 0 : static_cast<uint32_t>(rjetty->trans_mode),
                         rjetty == nullptr ? 0 : static_cast<uint32_t>(rjetty->tp_type),
                         cfg, target, target == nullptr ? -1 : 0);
        g_insideTrace = false;
    }
    return target;
}

extern "C" urma_status_t urma_bind_jetty(urma_jetty_t *jetty, urma_target_jetty_t *target)
{
    using Fn = urma_status_t (*)(urma_jetty_t *, urma_target_jetty_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_bind_jetty"));
    if (real == nullptr) return static_cast<urma_status_t>(-1);
    const urma_status_t status = real(jetty, target);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitJettyBind("urma_bind_jetty", jetty, target, nullptr, static_cast<int>(status));
        g_insideTrace = false;
    }
    return status;
}

extern "C" urma_status_t urma_bind_jetty_ex(urma_jetty_t *jetty,
    urma_target_jetty_t *target, urma_bind_jetty_ex_cfg_t *cfg)
{
    using Fn = urma_status_t (*)(urma_jetty_t *, urma_target_jetty_t *,
                                 urma_bind_jetty_ex_cfg_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_bind_jetty_ex"));
    if (real == nullptr) return static_cast<urma_status_t>(-1);
    const urma_status_t status = real(jetty, target, cfg);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitJettyBind("urma_bind_jetty_ex", jetty, target, cfg, static_cast<int>(status));
        g_insideTrace = false;
    }
    return status;
}
