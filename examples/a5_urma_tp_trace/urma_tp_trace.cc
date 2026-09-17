#include <dlfcn.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>

#include <urma_api.h>
#include <urma_cmd.h>

namespace {
std::mutex g_traceMutex;
thread_local bool g_insideTrace = false;

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
    std::lock_guard<std::mutex> guard(g_traceMutex);
    const char *prefix = std::getenv("A5_URMA_TP_TRACE_PREFIX");
    if (prefix == nullptr || prefix[0] == '\0') {
        std::fprintf(stderr, "%s\n", line.c_str());
        std::fflush(stderr);
        return;
    }
    const std::string path = std::string(prefix) + ".pid" + std::to_string(getpid()) + ".jsonl";
    FILE *file = std::fopen(path.c_str(), "a");
    if (file != nullptr) {
        std::fprintf(file, "%s\n", line.c_str());
        std::fclose(file);
    }
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

void EmitTpList(const char *api, const urma_get_tp_cfg_t *cfg, uint32_t requested,
                int status, uint32_t returned, const urma_tp_info_t *list)
{
    if (cfg == nullptr) return;
    std::ostringstream out;
    out << "{\"event\":\"GET_TP_LIST\",\"api\":\"" << api
        << "\",\"pid\":" << getpid() << ",\"status\":" << status
        << ",\"flag\":" << cfg->flag.value
        << ",\"trans_mode\":" << static_cast<uint32_t>(cfg->trans_mode)
        << ",\"requested\":" << requested << ",\"returned\":" << returned
        << ",\"local_eid_raw\":\"" << Hex(cfg->local_eid)
        << "\",\"peer_eid_raw\":\"" << Hex(cfg->peer_eid)
        << "\",\"tp_handles\":" << Handles(list, returned) << '}';
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
        out << ",\"attr_raw\":\"";
        const auto *bytes = reinterpret_cast<const unsigned char *>(attrs);
        static const char digits[] = "0123456789abcdef";
        for (size_t i = 0; i < static_cast<size_t>(count) * sizeof(*attrs); ++i) {
            out << digits[bytes[i] >> 4] << digits[bytes[i] & 0xf];
        }
        out << '"';
    }
    out << '}';
    Emit(out.str());
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
    out << '}';
    Emit(out.str());
}
} // namespace

extern "C" urma_status_t urma_get_tp_list(urma_context_t *ctx, urma_get_tp_cfg_t *cfg,
    uint32_t *tp_cnt, urma_tp_info_t *tp_list)
{
    using Fn = urma_status_t (*)(urma_context_t *, urma_get_tp_cfg_t *, uint32_t *, urma_tp_info_t *);
    static Fn real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "urma_get_tp_list"));
    if (real == nullptr) return static_cast<urma_status_t>(-1);
    const uint32_t requested = tp_cnt == nullptr ? 0 : *tp_cnt;
    const urma_status_t status = real(ctx, cfg, tp_cnt, tp_list);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitTpList("urma_get_tp_list", cfg, requested, static_cast<int>(status),
                   tp_cnt == nullptr ? 0 : *tp_cnt, tp_list);
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
    const int status = real(ctx, cfg, tp_cnt, tp_list, udata);
    if (!g_insideTrace) {
        g_insideTrace = true;
        EmitTpList("urma_cmd_get_tp_list", cfg, requested, status,
                   tp_cnt == nullptr ? 0 : *tp_cnt, tp_list);
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
