/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_MATRIX_MATMUL_SPARSE_OP_KERNEL_MATMUL_SPARSE_CUSTOM_KERNEL_H
#define EXAMPLES_MATRIX_MATMUL_SPARSE_OP_KERNEL_MATMUL_SPARSE_CUSTOM_KERNEL_H
#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"

using namespace matmul;

namespace MatmulSparseCustom {
template <typename ATYPE, typename BType, typename CType, typename BiasType>
class MatmulKernel {
public:
    __aicore__ inline MatmulKernel(){};
    /**
      * @brief  Initialization before process.
      * @param  a: A matrix gm addr.
      * @param  b: B matrix gm addr.
      * @param  bias: Bias matrix gm addr.
      * @param  index: Index matrix gm addr.
      * @param  c: C matrix gm addr.
      * @param  tiling: Matmul tiling struct.
      * @param  isTransA: Whether A matrix is transposed.
      * @param  isTransB: Whether B matrix is transposed.
      * @retval None
      */
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR index, GM_ADDR c, const TCubeTiling& tiling,
        bool isTransA, bool isTransB);
    /**
      * @brief  Process matrix calculation.
      * @retval None
      */
    __aicore__ inline void Process();

    using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, ATYPE, false>;
    // use SparseMatmulType to define B_TYPE
    using B_TYPE = AscendC::SparseMatmulType<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND, BType, true>;
    using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
    using BIAS_TYPE =  AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
    AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> matmulObj;

private:
    /**
      * @brief  Calculate the gm offset based on the blockIdx.
      * @param  blockIdx: Current Core blockidx.
      * @param  offsetA: Gm offset of A matrix.
      * @param  offsetB: Gm offset of B matrix.
      * @param  offsetC: Gm offset of C matrix.
      * @param  offsetBias: Gm offset of Bias matrix.
      * @param  offsetIndex: Gm offset of Index matrix.
      * @retval None
      */
    __aicore__ inline void CalcOffset(
        int32_t blockIdx, int32_t& offsetA, int32_t& offsetB, int32_t& offsetC, int32_t& offsetBias, int32_t& offsetIndex);

    AscendC::GlobalTensor<ATYPE> aGlobal;
    AscendC::GlobalTensor<BType> bGlobal;
    AscendC::GlobalTensor<CType> cGlobal;
    AscendC::GlobalTensor<BiasType> biasGlobal;
    AscendC::GlobalTensor<uint8_t> indexGlobal;
    TCubeTiling tiling;
    int32_t mCoreIndex;
    int32_t nCoreIndex;
    bool isTransA{false};
    bool isTransB{false};
};
} // namespace MatmulSparseCustom

#endif // EXAMPLES_MATRIX_MATMUL_SPARSE_OP_KERNEL_MATMUL_SPARSE_CUSTOM_KERNEL_H
