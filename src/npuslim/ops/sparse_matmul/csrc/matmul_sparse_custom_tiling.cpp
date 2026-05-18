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
#include <iostream>

namespace MatmulHost {

TCubeTiling GenerateTiling(const MatmulCaseParams& testCaseParams)
{
    TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(*ascendcPlatform);
    uint32_t M = testCaseParams.m;
    uint32_t N = testCaseParams.n;
    uint32_t K = testCaseParams.k;
    uint32_t blockDim = testCaseParams.usedCoreNum;
    bool isBias = testCaseParams.isBias;
    bool isAtrans = testCaseParams.isATrans;
    bool isBtrans = testCaseParams.isBTrans;

    cubeTiling.SetDim(blockDim);
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT8, isAtrans);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT8, isBtrans);
    cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT32);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT32);
    cubeTiling.SetSparse(true); // set sparse matmul
    cubeTiling.SetOrgShape(M, N, K);
    cubeTiling.SetShape(M, N, K);
    cubeTiling.EnableBias(isBias);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    if (cubeTiling.GetTiling(tilingData) == -1) {
        std::cout << "Generate tiling failed." << std::endl;
        return {};
    }
    return tilingData;
}

} // namespace MatmulHost
