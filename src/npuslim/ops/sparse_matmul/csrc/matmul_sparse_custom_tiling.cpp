/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "matmul_sparse_custom_tiling.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace MatmulHost {
namespace {

struct TilingAttempt {
    int32_t dim;
    bool splitK;
    int32_t maxSingleK;
    int32_t minSingleK;
    int32_t alignK;
};

struct TilingCacheKey {
    int32_t usedCoreNum;
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t isBias;
    bool isATrans;
    bool isBTrans;

    bool operator==(const TilingCacheKey& other) const
    {
        return usedCoreNum == other.usedCoreNum &&
               m == other.m &&
               n == other.n &&
               k == other.k &&
               isBias == other.isBias &&
               isATrans == other.isATrans &&
               isBTrans == other.isBTrans;
    }
};

struct TilingCacheKeyHash {
    std::size_t operator()(const TilingCacheKey& key) const
    {
        std::size_t h = 0;
        auto hash_combine = [&h](std::size_t value) {
            h ^= value + 0x9e3779b9 + (h << 6) + (h >> 2);
        };

        hash_combine(std::hash<int32_t>{}(key.usedCoreNum));
        hash_combine(std::hash<int32_t>{}(key.m));
        hash_combine(std::hash<int32_t>{}(key.n));
        hash_combine(std::hash<int32_t>{}(key.k));
        hash_combine(std::hash<int32_t>{}(key.isBias));
        hash_combine(std::hash<bool>{}(key.isATrans));
        hash_combine(std::hash<bool>{}(key.isBTrans));
        return h;
    }
};

struct TilingCacheEntry {
    bool success;
    TCubeTiling tilingData;
};

TilingCacheKey MakeCacheKey(const MatmulCaseParams& testCaseParams)
{
    return TilingCacheKey{
        testCaseParams.usedCoreNum,
        testCaseParams.m,
        testCaseParams.n,
        testCaseParams.k,
        testCaseParams.isBias,
        testCaseParams.isATrans,
        testCaseParams.isBTrans,
    };
}

std::unordered_map<TilingCacheKey, TilingCacheEntry, TilingCacheKeyHash>& GetTilingCache()
{
    static std::unordered_map<TilingCacheKey, TilingCacheEntry, TilingCacheKeyHash> cache;
    return cache;
}

std::mutex& GetTilingCacheMutex()
{
    static std::mutex mutex;
    return mutex;
}

void ConfigureCommonTiling(
    matmul_tiling::MultiCoreMatmulTiling& cubeTiling,
    const MatmulCaseParams& testCaseParams,
    int32_t blockDim)
{
    uint32_t M = testCaseParams.m;
    uint32_t N = testCaseParams.n;
    uint32_t K = testCaseParams.k;
    bool isBias = testCaseParams.isBias;
    bool isAtrans = testCaseParams.isATrans;
    bool isBtrans = testCaseParams.isBTrans;

    cubeTiling.SetDim(blockDim);
    cubeTiling.SetAType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT8,
        isAtrans);
    cubeTiling.SetBType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT8,
        isBtrans);
    cubeTiling.SetCType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT32);
    cubeTiling.SetBiasType(
        matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT32);
    cubeTiling.SetSparse(true);
    cubeTiling.SetOrgShape(M, N, K);
    cubeTiling.SetShape(M, N, K);
    cubeTiling.EnableBias(isBias);
    cubeTiling.SetBufferSpace(-1, -1, -1);
}

std::vector<int32_t> BuildDimCandidates(int32_t maxDim)
{
    std::vector<int32_t> dims;
    dims.reserve(static_cast<size_t>(maxDim));
    for (int32_t dim = maxDim; dim >= 1; --dim) {
        dims.push_back(dim);
    }
    return dims;
}

std::vector<TilingAttempt> BuildAttempts(const MatmulCaseParams& testCaseParams)
{
    std::vector<TilingAttempt> attempts;
    const auto dims = BuildDimCandidates(std::max(1, testCaseParams.usedCoreNum));
    const int32_t k = std::max(1, testCaseParams.k);

    auto appendAttempts = [&](bool splitK, int32_t maxSingleK, int32_t minSingleK, int32_t alignK) {
        for (int32_t dim : dims) {
            attempts.push_back(TilingAttempt{dim, splitK, maxSingleK, minSingleK, alignK});
        }
    };

    appendAttempts(false, -1, -1, -1);
    appendAttempts(true, -1, -1, -1);

    for (int32_t maxSingleK : {1024, 512, 256, 128}) {
        if (k > maxSingleK) {
            appendAttempts(false, maxSingleK, 64, 64);
            appendAttempts(true, maxSingleK, 64, 64);
        }
    }

    return attempts;
}

bool TryGenerateTiling(
    const MatmulCaseParams& testCaseParams,
    const TilingAttempt& attempt,
    TCubeTiling& tilingData)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(*ascendcPlatform);

    ConfigureCommonTiling(cubeTiling, testCaseParams, attempt.dim);

    if (attempt.splitK) {
        cubeTiling.EnableMultiCoreSplitK(true);
    }
    if (attempt.alignK > 0) {
        cubeTiling.SetAlignSplit(-1, -1, attempt.alignK);
    }
    if (attempt.maxSingleK > 0 || attempt.minSingleK > 0) {
        cubeTiling.SetSingleRange(
            -1,
            -1,
            attempt.maxSingleK,
            -1,
            -1,
            attempt.minSingleK);
    }

    return cubeTiling.GetTiling(tilingData) != -1;
}

} // namespace

bool GenerateTiling(const MatmulCaseParams& testCaseParams, TCubeTiling& tilingData)
{
    const auto cacheKey = MakeCacheKey(testCaseParams);
    {
        std::lock_guard<std::mutex> lock(GetTilingCacheMutex());
        const auto& cache = GetTilingCache();
        auto it = cache.find(cacheKey);
        if (it != cache.end()) {
            if (it->second.success) {
                tilingData = it->second.tilingData;
            }
            return it->second.success;
        }
    }

    for (const auto& attempt : BuildAttempts(testCaseParams)) {
        if (TryGenerateTiling(testCaseParams, attempt, tilingData)) {
            std::lock_guard<std::mutex> lock(GetTilingCacheMutex());
            GetTilingCache()[cacheKey] = TilingCacheEntry{true, tilingData};
            return true;
        }
    }

    {
        std::lock_guard<std::mutex> lock(GetTilingCacheMutex());
        GetTilingCache()[cacheKey] = TilingCacheEntry{false, {}};
    }
    return false;
}

} // namespace MatmulHost
