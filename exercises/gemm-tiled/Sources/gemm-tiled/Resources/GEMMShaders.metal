#include <metal_stdlib>

using namespace metal;

constant uint tileSize = 16;

struct GEMMUniforms {
    uint m;
    uint n;
    uint k;
};

kernel void tiled_gemm(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadPositionInGrid [[thread_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    threadgroup float tileA[tileSize][tileSize];
    threadgroup float tileB[tileSize][tileSize];

    uint row = threadPositionInGrid.y;
    uint col = threadPositionInGrid.x;
    float accumulator = 0.0f;

    for (uint tileStart = 0; tileStart < uniforms.k; tileStart += tileSize) {
        uint aCol = tileStart + threadPositionInThreadgroup.x;
        uint bRow = tileStart + threadPositionInThreadgroup.y;

        tileA[threadPositionInThreadgroup.y][threadPositionInThreadgroup.x] =
            (row < uniforms.m && aCol < uniforms.k) ? a[row * uniforms.k + aCol] : 0.0f;
        tileB[threadPositionInThreadgroup.y][threadPositionInThreadgroup.x] =
            (bRow < uniforms.k && col < uniforms.n) ? b[bRow * uniforms.n + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < tileSize; inner++) {
            accumulator += tileA[threadPositionInThreadgroup.y][inner] * tileB[inner][threadPositionInThreadgroup.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < uniforms.m && col < uniforms.n) {
        c[row * uniforms.n + col] = accumulator;
    }
}
