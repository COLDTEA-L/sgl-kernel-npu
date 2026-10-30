#ifndef A5_TEST_FAKE_CCU_KERNEL_H
#define A5_TEST_FAKE_CCU_KERNEL_H
// CPU representation model only. Never used to build the device operator.
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
using ChannelHandle = uint64_t;
using HcclResult = int;
constexpr int HCCL_SUCCESS = 0, HCCL_E_PARA = 1;
namespace hcomm {
namespace CcuRep {
struct Env { std::vector<uint64_t> args; std::function<uint64_t(ChannelHandle, uint32_t)> remote; };
using Expr = std::function<uint64_t(const Env&)>;
inline std::vector<Expr>& Conditions() { static thread_local std::vector<Expr> stack; return stack; }
// Conditional register assignments form a DAG. Memoize within ONE expression
// evaluation, never across instructions/epochs where remote memory may change.
inline Expr Memo(Expr expression) {
    static size_t next=0;
    const size_t id=++next;
    const auto shared=std::make_shared<Expr>(std::move(expression));
    return [id,shared](const Env& e) {
        static thread_local size_t depth=0;
        static thread_local std::unordered_map<size_t,uint64_t> values;
        if(depth==0) values.clear();
        auto found=values.find(id);
        if(found!=values.end()) return found->second;
        ++depth;
        uint64_t value;
        try { value=(*shared)(e); } catch(...) {--depth;throw;}
        --depth; values[id]=value;return value;
    };
}
struct Variable {
    std::function<uint64_t(const Env &)> eval = [](const Env &) { return 0ULL; };
    Variable()=default;
    Variable(const Variable&)=default;
    Variable& Assign(Expr value) {
        const auto old=eval;
        const auto guards=Conditions();
        eval=Memo([old,value,guards](const Env& e) {
            for(const auto& guard:guards) if(!guard(e)) return old(e);
            return value(e);
        }); return *this;
    }
    Variable& operator=(const Variable& other) { return Assign(other.eval); }
    Variable &operator=(uint64_t value) { return Assign([value](const Env&) {return value;}); }
    Variable &operator+=(const Variable &other) {
        const auto left = eval, right = other.eval;
        return Assign([left, right](const Env &e) { return left(e) + right(e); });
    }
};
inline Variable operator!=(const Variable &v, uint64_t value) {
    Variable result; const auto expr = v.eval;
    result.eval = [expr, value](const Env &e) { return uint64_t(expr(e) != value); }; return result;
}
inline Variable operator==(const Variable& v,uint64_t value) {
    Variable out;auto a=v.eval;out.eval=[a,value](const Env& e){return uint64_t(a(e)==value);};return out;
}
inline Variable operator-(const Variable& a,const Variable& b) {
#ifdef A5_TEST_CCU_V1_ONLY
    (void)a;(void)b;throw std::runtime_error("V1 unsupported subtraction repType[11]");
#else
    Variable out;auto x=a.eval,y=b.eval;out.eval=[x,y](const Env& e){return x(e)-y(e);};return out;
#endif
}
inline Variable operator&(const Variable& a,const Variable& b) {
#ifdef A5_TEST_CCU_V1_ONLY
    (void)a;(void)b;throw std::runtime_error("V1 unsupported bitwise AND");
#else
    Variable out;auto x=a.eval,y=b.eval;out.eval=[x,y](const Env& e){return x(e)&y(e);};return out;
#endif
}
inline Variable operator>=(const Variable& a,uint64_t value) {
    Variable out;auto x=a.eval;out.eval=[x,value](const Env& e){return uint64_t(x(e)>=value);};return out;
}
struct LocalAddr { Variable addr, token; };
struct RemoteAddr { Variable addr, token; };
inline size_t& EventAllocations() {static size_t count=0;return count;}
struct CompletedEvent {
    int id;
    explicit CompletedEvent(int value=-1):id(value) {++EventAllocations();}
};
} // namespace CcuRep
struct CcuKernelSignature {
    std::string value;
    void Append(const std::string &s) { value += s; }
};
struct CcuKernelArg {
    std::vector<ChannelHandle> channels;
    virtual ~CcuKernelArg() = default;
    virtual CcuKernelSignature GetKernelSignature() const = 0;
};
struct CcuTaskArg { virtual ~CcuTaskArg() = default; };
struct Instruction {
    std::string op;
    ChannelHandle channel = 0;
    uint32_t slot = 0;
    uint32_t mask = 0, variable = UINT32_MAX;
    int event = -1;
    CcuRep::Variable dst, src, len;
};
class CcuKernel {
public:
    explicit CcuKernel(const CcuKernelArg &arg) : channels_(arg.channels) {}
    virtual ~CcuKernel() = default;
    HcclResult Init() { return Algorithm(); }
    std::vector<Instruction> instructions;
protected:
    std::vector<ChannelHandle> channels_;
    size_t loadIndex_ = 0;
    int nextEvent_ = 0;
    virtual HcclResult Algorithm() = 0;
    virtual std::vector<uint64_t> GeneArgs(const CcuTaskArg &) = 0;
    // Record both branches symbolically; the scheduler evaluates each launch.
    struct ConditionalScope {
        CcuKernel *owner;
        ConditionalScope(CcuKernel *kernel, const CcuRep::Variable &v) : owner(kernel) {
            Instruction i; i.op="if_begin"; i.len=v; owner->instructions.push_back(i);
            CcuRep::Conditions().push_back(v.eval);
        }
        ConditionalScope(ConditionalScope &&other) : owner(other.owner) { other.owner=nullptr; }
        ConditionalScope(const ConditionalScope &) = delete;
        explicit operator bool() const { return true; }
        ~ConditionalScope() { if(owner) { CcuRep::Conditions().pop_back(); Instruction i; i.op="if_end"; owner->instructions.push_back(i); } }
    };
    ConditionalScope Conditional(const CcuRep::Variable &v) { return ConditionalScope(this,v); }
    CcuRep::Variable CreateVariable() { return {}; }
    CcuRep::Variable BorrowChannelVariable(ChannelHandle ch, uint32_t slot) {
        CcuRep::Variable v;
        v.eval = [ch, slot](const CcuRep::Env &e) { return e.remote(ch, slot); };return v;
    }
    HcclResult CreateVariable(ChannelHandle ch, uint32_t slot, CcuRep::Variable *v) {
        v->eval = [ch, slot](const CcuRep::Env &e) { return e.remote(ch, slot); }; return 0;
    }
    void Load(CcuRep::Variable &v) {
        const size_t index = loadIndex_++;
        v.eval = [index](const CcuRep::Env &e) { return e.args.at(index); };
    }
    CcuRep::LocalAddr CreateLocalAddr() { return {}; }
    CcuRep::RemoteAddr CreateRemoteAddr() { return {}; }
    CcuRep::CompletedEvent CreateCompletedEvent() { return CcuRep::CompletedEvent(nextEvent_++); }
    HcclResult NotifyRecord(ChannelHandle ch, uint32_t slot, uint32_t variable,
        const CcuRep::Variable &value, uint32_t mask) {
        Instruction i; i.op="record"; i.channel=ch; i.slot=slot; i.mask=mask; i.variable=variable; i.len=value;
        instructions.push_back(i); return 0;
    }
    HcclResult NotifyRecord(ChannelHandle ch, uint32_t slot, uint32_t mask) {
        Instruction i; i.op = "record"; i.channel = ch; i.slot = slot; i.mask=mask; instructions.push_back(i); return 0;
    }
    HcclResult NotifyWait(ChannelHandle ch, uint32_t slot, uint32_t mask) {
        Instruction i; i.op = "notify_wait"; i.channel = ch; i.slot = slot; i.mask=mask; instructions.push_back(i); return 0;
    }
    HcclResult LocalCopyNb(CcuRep::LocalAddr dst, CcuRep::LocalAddr src,
        CcuRep::Variable len, CcuRep::CompletedEvent e) {
        return Copy("local", 0, dst.addr, src.addr, len, e);
    }
    HcclResult WriteNb(ChannelHandle ch, CcuRep::RemoteAddr dst, CcuRep::LocalAddr src,
        CcuRep::Variable len, CcuRep::CompletedEvent e) {
        return Copy("remote", ch, dst.addr, src.addr, len, e);
    }
    HcclResult Copy(const char *op, ChannelHandle ch, CcuRep::Variable dst,
        CcuRep::Variable src, CcuRep::Variable len, CcuRep::CompletedEvent e) {
        Instruction i; i.op=op; i.channel=ch; i.dst=dst; i.src=src; i.len=len; i.event=e.id;
        instructions.push_back(i); return 0;
    }
    HcclResult WaitEvent(CcuRep::CompletedEvent e) {
        Instruction i; i.op="event_wait"; i.event=e.id; instructions.push_back(i); return 0;
    }
};
} // namespace hcomm
// Mirrors the SDK graph-builder conditional, not a C++ data-dependent branch.
#define CCU_IF(condition) if (auto ccu_condition = this->Conditional(condition))
#endif
