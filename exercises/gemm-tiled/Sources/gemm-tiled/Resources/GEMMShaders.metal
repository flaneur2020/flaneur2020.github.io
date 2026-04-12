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
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    threadgroup float tileA[16][16];
    threadgroup float tileB[16][16];

    uint row = threadgroupPosition.y * 16 + threadPositionInThreadgroup.y;
    uint col = threadgroupPosition.x * 16 + threadPositionInThreadgroup.x;
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
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    threadgroup float tileA[32][32];
    threadgroup float tileB[32][32];

    uint row = threadgroupPosition.y * 32 + threadPositionInThreadgroup.y;
    uint col = threadgroupPosition.x * 32 + threadPositionInThreadgroup.x;
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
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    threadgroup float tileA[32][32];
    threadgroup float tileB[32][32];

    uint row = threadgroupPosition.y * 32 + threadPositionInThreadgroup.y;
    uint col = threadgroupPosition.x * 32 + threadPositionInThreadgroup.x;
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

kernel void register_blocked_gemm_4x4(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 32;
    constexpr uint blockN = 32;
    constexpr uint blockK = 8;
    constexpr uint registerBlockM = 4;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 8;

    threadgroup float tileA[blockM][blockK];
    threadgroup float tileB[blockK][blockN];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localRowBase = threadPositionInThreadgroup.y * registerBlockM;
    uint localColBase = threadPositionInThreadgroup.x * registerBlockN;
    uint globalRowBase = threadgroupPosition.y * blockM + localRowBase;
    uint globalColBase = threadgroupPosition.x * blockN + localColBase;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    for (uint tileStart = 0; tileStart < uniforms.k; tileStart += blockK) {
        for (uint loadIndex = localLinearIndex; loadIndex < blockM * blockK; loadIndex += 64) {
            uint rowIndex = loadIndex / blockK;
            uint innerIndex = loadIndex % blockK;
            uint globalACol = tileStart + innerIndex;
            uint globalARow = threadgroupPosition.y * blockM + rowIndex;
            tileA[rowIndex][innerIndex] =
                (globalARow < uniforms.m && globalACol < uniforms.k) ? a[globalARow * uniforms.k + globalACol] : 0.0f;
        }

        for (uint loadIndex = localLinearIndex; loadIndex < blockK * blockN; loadIndex += 64) {
            uint innerIndex = loadIndex / blockN;
            uint colIndex = loadIndex % blockN;
            uint globalBRow = tileStart + innerIndex;
            uint globalBCol = threadgroupPosition.x * blockN + colIndex;
            tileB[innerIndex][colIndex] =
                (globalBRow < uniforms.k && globalBCol < uniforms.n) ? b[globalBRow * uniforms.n + globalBCol] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < blockK; inner++) {
            float4 bVector = float4(
                tileB[inner][localColBase + 0],
                tileB[inner][localColBase + 1],
                tileB[inner][localColBase + 2],
                tileB[inner][localColBase + 3]
            );

            accumulator0 += tileA[localRowBase + 0][inner] * bVector;
            accumulator1 += tileA[localRowBase + 1][inner] * bVector;
            accumulator2 += tileA[localRowBase + 2][inner] * bVector;
            accumulator3 += tileA[localRowBase + 3][inner] * bVector;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (globalRowBase + 0 < uniforms.m) {
        if (globalColBase + 0 < uniforms.n) { c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0]; }
        if (globalColBase + 1 < uniforms.n) { c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1]; }
        if (globalColBase + 2 < uniforms.n) { c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2]; }
        if (globalColBase + 3 < uniforms.n) { c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3]; }
    }
    if (globalRowBase + 1 < uniforms.m) {
        if (globalColBase + 0 < uniforms.n) { c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0]; }
        if (globalColBase + 1 < uniforms.n) { c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1]; }
        if (globalColBase + 2 < uniforms.n) { c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2]; }
        if (globalColBase + 3 < uniforms.n) { c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3]; }
    }
    if (globalRowBase + 2 < uniforms.m) {
        if (globalColBase + 0 < uniforms.n) { c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0]; }
        if (globalColBase + 1 < uniforms.n) { c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1]; }
        if (globalColBase + 2 < uniforms.n) { c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2]; }
        if (globalColBase + 3 < uniforms.n) { c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3]; }
    }
    if (globalRowBase + 3 < uniforms.m) {
        if (globalColBase + 0 < uniforms.n) { c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0]; }
        if (globalColBase + 1 < uniforms.n) { c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1]; }
        if (globalColBase + 2 < uniforms.n) { c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2]; }
        if (globalColBase + 3 < uniforms.n) { c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3]; }
    }
}
