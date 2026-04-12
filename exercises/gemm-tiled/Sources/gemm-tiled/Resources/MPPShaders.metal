#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void mpp_matmul_cooperative_f32_64x32(
    tensor<device float, dextents<int32_t, 2>> aTensor,
    tensor<device float, dextents<int32_t, 2>> bTensor,
    tensor<device float, dextents<int32_t, 2>> cTensor
) {
    constexpr auto matmulDescriptor = matmul2d_descriptor(
        64,
        32,
        static_cast<int>(dynamic_extent)
    );

    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;
    matmulOp.run(aTensor, bTensor, cTensor);
}
