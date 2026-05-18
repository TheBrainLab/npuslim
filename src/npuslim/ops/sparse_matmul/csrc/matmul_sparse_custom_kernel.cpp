/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "matmul_sparse_custom_kernel.h"

namespace {
/**
  * @brief  Copy tiling data to TCubeTiling ptr from tiling gm addr.
  * @param  tiling: TCubeTiling ptr which needs to copy tiling data.
  * @param  tilingGM: Tiling gm addr.
  * @retval None
  */
__aicore__ inline void CopyTiling(TCubeTiling* tiling, GM_ADDR tilingGM)
{
    uint32_t* ptr = reinterpret_cast<uint32_t*>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t*>(tilingGM);

    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
      *ptr = *(tiling32 + i);
    }
    return;
}
}

namespace MatmulSparseCustom {

constexpr int32_t DENSE_MATRIX_B_OFFSET = 2;
constexpr int32_t INDEX_MATRIX_OFFSET = 8;

template <typename AType, typename BType, typename CType, typename BiasType>
__aicore__ inline void MatmulKernel<AType, BType, CType, BiasType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias,
    GM_ADDR index, GM_ADDR c, const TCubeTiling& tiling, bool isTransA, bool isTransB)
{
    this->tiling = tiling;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ AType*>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BType*>(b), tiling.Kb / DENSE_MATRIX_B_OFFSET * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ CType*>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType*>(bias), tiling.N);
    indexGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(index), tiling.N * tiling.Kb / INDEX_MATRIX_OFFSET);

    int32_t offsetA = 0;
    int32_t offsetB = 0;
    int32_t offsetC = 0;
    int32_t offsetBias = 0;
    int32_t offsetIndex = 0;
    this->isTransA = isTransA;
    this->isTransB = isTransB;
    CalcOffset(AscendC::GetBlockIdx(), offsetA, offsetB, offsetC, offsetBias, offsetIndex);
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];
    indexGlobal = indexGlobal[offsetIndex];
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}

template <typename AType, typename BType, typename CType, typename BiasType>
__aicore__ inline void MatmulKernel<AType, BType, CType, BiasType>::Process()
{
    if (matmul::GetBlockIdx() >= tiling.usedCoreNum) {
        return;
    }

    // process with tail block
    int tailM = tiling.M - mCoreIndex * tiling.singleCoreM;
    tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
    int tailN = tiling.N - nCoreIndex * tiling.singleCoreN;
    tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
    if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN) {
        matmulObj.SetTail(tailM, tailN);
    }

    matmulObj.SetTensorA(aGlobal, isTransA);
    matmulObj.SetTensorB(bGlobal, isTransB);
    matmulObj.SetSparseIndex(indexGlobal); // set index matrix
    if (tiling.isBias) {
        matmulObj.SetBias(biasGlobal);
    }

    matmulObj.IterateAll(cGlobal);
    matmulObj.End();
}

template <typename AType, typename BType, typename CType, typename BiasType>
__aicore__ inline void MatmulKernel<AType, BType, CType, BiasType>::CalcOffset(
    int32_t blockIdx, int32_t& offsetA, int32_t& offsetB, int32_t& offsetC, int32_t& offsetBias, int32_t& offsetIndex)
{
    const TCubeTiling& tiling = this->tiling;
    auto mSingleBlocks = (tiling.M + tiling.singleCoreM - 1) / tiling.singleCoreM; // split M into mSingleBlocks cores
    mCoreIndex = blockIdx % mSingleBlocks;
    nCoreIndex = blockIdx / mSingleBlocks;

    offsetA = mCoreIndex * tiling.Ka * tiling.singleCoreM;
    if (isTransA) {
        offsetA = mCoreIndex * tiling.singleCoreM;
    }
    offsetB = nCoreIndex * tiling.singleCoreN;
    if (isTransB) {
        offsetB = nCoreIndex * tiling.Kb * tiling.singleCoreN;
    }
    offsetB = offsetB / DENSE_MATRIX_B_OFFSET;
    offsetC = mCoreIndex * tiling.N * tiling.singleCoreM + nCoreIndex * tiling.singleCoreN;
    offsetBias = nCoreIndex * tiling.singleCoreN;
    offsetIndex = nCoreIndex * tiling.singleCoreN * INDEX_MATRIX_OFFSET;
}
} // namespace MatmulSparseCustom

/**
  * @brief  matmul kernel function.
  * @param  a: A matrix gm addr.
  * @param  b: B matrix gm addr.
  * @param  bias: Bias matrix gm addr.
  * @param  index: Index matrix gm addr.
  * @param  c: C matrix gm addr.
  * @param  workspace: Temporary gm space addr required by matmul calc.
  * @param  tilingGm: Tiling data addr. 
  * @retval None
  */
extern "C" __global__ __aicore__ void matmul_sparse_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR index, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm)
{
    if (g_coreType == AscendC::AIV) {
        return;
    }
    // prepare tiling
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);
    // define matmul kernel
    MatmulSparseCustom::MatmulKernel<int8_t, int8_t, int32_t, int32_t> matmulKernel;
    AscendC::TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulKernel.matmulObj, &tiling);
    // init matmul kernel, isTransA=false, isTransB=true
    matmulKernel.Init(a, b, bias, index, c, tiling, false, true);
    // matmul kernel process
    matmulKernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void matmul_sparse_custom_do(uint32_t blockDim, void* stream,
    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR index, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm)
{
    matmul_sparse_custom<<<blockDim, nullptr, stream>>>(a, b, bias, index, c, workspace, tilingGm);
}
#endif
