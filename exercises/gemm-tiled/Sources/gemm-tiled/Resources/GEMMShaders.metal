#include <metal_stdlib>

using namespace metal;

struct GEMMUniforms {
    uint m;
    uint n;
    uint k;
};

kernel void tiled_gemm_16x16(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadPositionInGrid [[thread_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    threadgroup float tileA[16][16];
    threadgroup float tileB[16][16];

    uint row = threadPositionInGrid.y;
    uint col = threadPositionInGrid.x;
    float accumulator = 0.0f;

    for (uint tileStart = 0; tileStart < uniforms.k; tileStart += 16) {
        uint aCol = tileStart + threadPositionInThreadgroup.x;
        uint bRow = tileStart + threadPositionInThreadgroup.y;

        tileA[threadPositionInThreadgroup.y][threadPositionInThreadgroup.x] =
            (row < uniforms.m && aCol < uniforms.k) ? a[row * uniforms.k + aCol] : 0.0f;
        tileB[threadPositionInThreadgroup.y][threadPositionInThreadgroup.x] =
            (bRow < uniforms.k && col < uniforms.n) ? b[bRow * uniforms.n + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < 16; inner++) {
            accumulator += tileA[threadPositionInThreadgroup.y][inner] * tileB[inner][threadPositionInThreadgroup.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < uniforms.m && col < uniforms.n) {
        c[row * uniforms.n + col] = accumulator;
    }
}

kernel void tiled_gemm_32x32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadPositionInGrid [[thread_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    threadgroup float tileA[32][32];
    threadgroup float tileB[32][32];

    uint row = threadPositionInGrid.y;
    uint col = threadPositionInGrid.x;
    float accumulator = 0.0f;

    for (uint tileStart = 0; tileStart < uniforms.k; tileStart += 32) {
        uint aCol = tileStart + threadPositionInThreadgroup.x;
        uint bRow = tileStart + threadPositionInThreadgroup.y;

        tileA[threadPositionInThreadgroup.y][threadPositionInThreadgroup.x] =
            (row < uniforms.m && aCol < uniforms.k) ? a[row * uniforms.k + aCol] : 0.0f;
        tileB[threadPositionInThreadgroup.y][threadPositionInThreadgroup.x] =
            (bRow < uniforms.k && col < uniforms.n) ? b[bRow * uniforms.n + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < 32; inner++) {
            accumulator += tileA[threadPositionInThreadgroup.y][inner] * tileB[inner][threadPositionInThreadgroup.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < uniforms.m && col < uniforms.n) {
        c[row * uniforms.n + col] = accumulator;
    }
}

kernel void swizzled_gemm_32x32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadPositionInGrid [[thread_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    threadgroup float tileA[32][32];
    threadgroup float tileB[32][32];

    uint row = threadPositionInGrid.y;
    uint col = threadPositionInGrid.x;
    uint localX = threadPositionInThreadgroup.x;
    uint localY = threadPositionInThreadgroup.y;
    float accumulator = 0.0f;

    for (uint tileStart = 0; tileStart < uniforms.k; tileStart += 32) {
        uint aCol = tileStart + localX;
        uint bRow = tileStart + localY;
        uint swizzledColumn = localX ^ (localY & 7);

        tileA[localY][localX] =
            (row < uniforms.m && aCol < uniforms.k) ? a[row * uniforms.k + aCol] : 0.0f;
        tileB[localY][swizzledColumn] =
            (bRow < uniforms.k && col < uniforms.n) ? b[bRow * uniforms.n + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < 32; inner++) {
            uint swizzledReadColumn = localX ^ (inner & 7);
            accumulator += tileA[localY][inner] * tileB[inner][swizzledReadColumn];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < uniforms.m && col < uniforms.n) {
        c[row * uniforms.n + col] = accumulator;
    }
}
