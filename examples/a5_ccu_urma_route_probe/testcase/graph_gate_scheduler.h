#ifndef A5_GRAPH_GATE_SCHEDULER_H
#define A5_GRAPH_GATE_SCHEDULER_H

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Invoked by the normal host thread, NEVER inside an ACL host callback.
// One connection per rank; all participants receive one common plan/ticket.
class GateSchedulerClient {
public:
    explicit GateSchedulerClient(unsigned rank)
    {
        const char *path = std::getenv("A5_GATE_SCHED_SOCKET");
        if (!path) return;
        const char *job = std::getenv("A5_GATE_SCHED_JOB");
        if (!job || !*job || std::string(job).find_first_of(" \t\r\n") != std::string::npos)
            throw std::runtime_error("scheduler job missing/invalid");
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        if (std::strlen(path) >= sizeof(address.sun_path)) throw std::runtime_error("scheduler socket path too long");
        std::strcpy(address.sun_path, path);
        fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd_ < 0) Fail("socket");
        timeval timeout{};
        const char *seconds = std::getenv("A5_GATE_SCHED_TIMEOUT_S");
        timeout.tv_sec = seconds ? std::stol(seconds) : 300;
        if (timeout.tv_sec < 1 || timeout.tv_sec > 86400) throw std::runtime_error("invalid scheduler timeout");
        if (setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) ||
            setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout))) Fail("setsockopt");
        if (connect(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address))) Fail("connect");
        Send(std::string("HELLO ") + job + " " + std::to_string(rank));
        if (Receive() != "OK") throw std::runtime_error("scheduler HELLO rejected");
    }
    ~GateSchedulerClient() { if (fd_ >= 0) close(fd_); }
    GateSchedulerClient(const GateSchedulerClient &) = delete;
    GateSchedulerClient &operator=(const GateSchedulerClient &) = delete;
    bool Enabled() const { return fd_ >= 0; }

    std::vector<uint64_t> Arrive(unsigned iteration, unsigned op, size_t paths, size_t bytes)
    {
        Send("ARRIVED " + std::to_string(iteration) + " " + std::to_string(op));
        std::istringstream response(Receive());
        std::string command, extra;
        size_t count = 0;
        if (!(response >> command >> ticket_ >> count) || command != "GO" || !ticket_ || count != paths)
            throw std::runtime_error("invalid scheduler grant");
        std::vector<uint64_t> split(paths);
        uint64_t sum = 0;
        for (auto &value : split) {
            if (!(response >> value) || value > bytes || value % 4 || sum > bytes - value)
                throw std::runtime_error("invalid scheduler split");
            sum += value;
        }
        if (sum != bytes || response >> extra) throw std::runtime_error("scheduler split size mismatch");
        return split;
    }
    void Done(unsigned iteration, unsigned op, uint64_t releaseToDoneNs)
    {
        Send("DONE " + std::to_string(iteration) + " " + std::to_string(op) + " " +
             std::to_string(ticket_) + " " + std::to_string(releaseToDoneNs));
        if (Receive() != "ACK") throw std::runtime_error("scheduler DONE rejected");
    }
    uint64_t Ticket() const { return ticket_; }

private:
    int fd_ = -1;
    uint64_t ticket_ = 0;
    std::string received_;
    [[noreturn]] static void Fail(const char *operation)
    {
        throw std::runtime_error(std::string("scheduler ") + operation + ": " + std::strerror(errno));
    }
    void Send(const std::string &message)
    {
        const std::string data = message + "\n";
        size_t offset = 0;
        while (offset < data.size()) {
            const auto count = send(fd_, data.data() + offset, data.size() - offset, MSG_NOSIGNAL);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) Fail("send");
            offset += static_cast<size_t>(count);
        }
    }
    std::string Receive()
    {
        while (true) {
            const auto newline = received_.find('\n');
            if (newline != std::string::npos) {
                const auto line = received_.substr(0, newline);
                received_.erase(0, newline + 1);
                return line;
            }
            if (received_.size() >= 4096) throw std::runtime_error("scheduler response too long");
            char block[4096];
            const auto count = recv(fd_, block, sizeof(block) - received_.size(), 0);
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) Fail("receive");
            if (!count) throw std::runtime_error("scheduler disconnected; gate remains closed");
            received_.append(block, static_cast<size_t>(count));
        }
    }
};

inline unsigned GatePollUs()
{
    static const unsigned value = [] {
        const char *raw = std::getenv("A5_GATE_POLL_US");
        const auto result = raw ? std::stoul(raw) : 1000UL;
        if (result < 1 || result > 1000000) throw std::runtime_error("invalid gate poll interval");
        return static_cast<unsigned>(result);
    }();
    return value;
}
#endif
