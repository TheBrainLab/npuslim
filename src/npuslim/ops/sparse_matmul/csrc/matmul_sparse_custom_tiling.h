/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_MATRIX_MATMUL_SPARSE_OP_HOST_MATMUL_SPARSE_CUSTOM_TILING_H
#define EXAMPLES_MATRIX_MATMUL_SPARSE_OP_HOST_MATMUL_SPARSE_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace MatmulHost {

struct MatmulCaseParams
{
    int32_t usedCoreNum;
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t isBias;
    bool isATrans;
    bool isBTrans;
};

/**
  * @brief Generate matmul tiling.
  * @param testCaseParams: Testcase parameters.
  * @param tilingData: Output tiling data.
  * @retval true if tiling was generated successfully, false otherwise.
  */
bool GenerateTiling(const MatmulCaseParams& testCaseParams, TCubeTiling& tilingData);

} // namespace MatmulHost
#endif // EXAMPLES_MATRIX_MATMUL_SPARSE_OP_HOST_MATMUL_SPARSE_CUSTOM_TILING_H
