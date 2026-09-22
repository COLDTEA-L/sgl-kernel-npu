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
        // Comma-separated positive integers.  Kept as a string because the
        // public OpDef API in the supported CANN builds has no portable list
        // attribute ABI.
        this->Attr("path_weights").AttrType(REQUIRED).String();

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
