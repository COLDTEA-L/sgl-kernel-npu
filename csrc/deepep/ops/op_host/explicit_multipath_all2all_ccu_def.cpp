#include "register/op_def_registry.h"

namespace ops {
class ExplicitMultipathAll2AllCcu : public OpDef {
public:
    explicit ExplicitMultipathAll2AllCcu(const char *name) : OpDef(name)
    {
        this->Input("sendData")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        // One int64 policy word.  Nibble i is the runtime weight of path i;
        // zero disables that path.  The top nibble is the policy ABI.
        this->Input("pathPolicy")
            .ParamType(REQUIRED)
            // Code generation zips dtype/format alternatives across every
            // input.  Repeat the invariant policy type once for each
            // sendData alternative so alternatives 1..3 do not acquire a
            // zero-sized dtype in opbuild.
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("recvData")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("rank_size").AttrType(REQUIRED).Int();
        this->Attr("rank_id").AttrType(REQUIRED).Int();
        // plan_id is a stable control-plane key.  The graph never contains a
        // manifest pathname and never creates channels during replay.
        this->Attr("plan_id").AttrType(REQUIRED).String();

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend950", config);
        this->MC2().HcclGroup("group");
    }
};

OP_ADD(ExplicitMultipathAll2AllCcu);
}  // namespace ops
