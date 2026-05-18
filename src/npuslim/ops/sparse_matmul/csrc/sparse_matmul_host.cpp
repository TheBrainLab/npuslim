#include <cstdint>
#include <cstring>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel_tiling/kernel_tiling.h"
#include "matmul_sparse_custom_tiling.h"

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
    void* stream)
{
    // Generate tiling
    MatmulHost::MatmulCaseParams params{
        GetCoreNum(),
        static_cast<int32_t>(m),
        static_cast<int32_t>(n),
        static_cast<int32_t>(k),
        /*isBias=*/0, /*isATrans=*/false, /*isBTrans=*/true
    };
    TCubeTiling tilingData = MatmulHost::GenerateTiling(params);

    // Upload tiling to device
    size_t tilingSize = sizeof(TCubeTiling);
    void* tilingHost = nullptr;
    void* tilingDevice = nullptr;
    aclError ret;

    ret = aclrtMallocHost(&tilingHost, tilingSize);
    if (ret != ACL_SUCCESS) return -1;
    memcpy(tilingHost, &tilingData, tilingSize);

    ret = aclrtMalloc(&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { aclrtFreeHost(tilingHost); return -1; }

    ret = aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { aclrtFree(tilingDevice); aclrtFreeHost(tilingHost); return -1; }

    // Allocate workspace
    size_t workspaceSize = GetSysWorkSpaceSize();
    void* workspaceDev = nullptr;
    ret = aclrtMalloc(&workspaceDev, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { aclrtFree(tilingDevice); aclrtFreeHost(tilingHost); return -1; }

    // Launch kernel
    aclrtlaunch_matmul_sparse_custom(
        static_cast<uint32_t>(tilingData.usedCoreNum),
        static_cast<aclrtStream>(stream),
        a_dev, b_dev, /*bias=*/nullptr, index_dev,
        c_dev, workspaceDev, tilingDevice);

    aclrtFree(workspaceDev);
    aclrtFree(tilingDevice);
    aclrtFreeHost(tilingHost);
    return 0;
}
