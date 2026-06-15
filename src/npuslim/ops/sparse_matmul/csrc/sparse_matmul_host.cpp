#include <cstdint>
#include <cstring>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel_tiling/kernel_tiling.h"
#include "matmul_sparse_custom_tiling.h"

#include "torch_npu/csrc/framework/OpCommand.h"

extern "C" uint32_t aclrtlaunch_matmul_sparse_custom(
    uint32_t blockDim, aclrtStream stream,
    void* a, void* b, void* bias, void* index,
    void* c, void* workspace, void* tilingGm);

static size_t GetSysWorkSpaceSize()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    return static_cast<size_t>(platform->GetLibApiWorkSpaceSize());
}

static int32_t GetCoreNum()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    return static_cast<int32_t>(platform->GetCoreNumAic());
}

extern "C" int run_sparse_matmul(
    void* a_dev,       // [M, K] int8, device ptr
    void* b_dev,       // [N, K/2] int8, device ptr (densified sparse B)
    void* index_dev,   // [N_padded, K/8] uint8 NZ-format, device ptr
    void* c_dev,       // [M, N] int32, device ptr (output)
    int64_t m, int64_t n, int64_t k,
    void* workspace_dev,
    int64_t workspace_size,
    void* tiling_dev,
    int64_t tiling_size,
    void* stream)
{
    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    MatmulHost::MatmulCaseParams params{
        GetCoreNum(),
        static_cast<int32_t>(m),
        static_cast<int32_t>(n),
        static_cast<int32_t>(k),
        /*isBias=*/0, /*isATrans=*/false, /*isBTrans=*/true
    };
    TCubeTiling tilingData {};
    if (!MatmulHost::GenerateTiling(params, tilingData)) {
        return -1;
    }
    if (tiling_dev == nullptr ||
        tiling_size < static_cast<int64_t>(sizeof(TCubeTiling))) {
        return -1;
    }
    if (workspace_dev == nullptr ||
        workspace_size < static_cast<int64_t>(GetSysWorkSpaceSize())) {
        return -1;
    }
    aclError ret = aclrtMemcpy(
        tiling_dev,
        static_cast<size_t>(tiling_size),
        &tilingData,
        sizeof(TCubeTiling),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return -1;
    }

    at_npu::native::OpCommand::RunOpApiV2(
        "npuslim::sparse_matmul_4to2",
        [=]() -> int {
            aclError launchRet = aclrtlaunch_matmul_sparse_custom(
                static_cast<uint32_t>(tilingData.usedCoreNum),
                aclStream,
                a_dev, b_dev, /*bias=*/nullptr, index_dev,
                c_dev, workspace_dev, tiling_dev);
            return launchRet == ACL_SUCCESS ? 0 : -1;
        },
        /*sync=*/false);
    return 0;
}
