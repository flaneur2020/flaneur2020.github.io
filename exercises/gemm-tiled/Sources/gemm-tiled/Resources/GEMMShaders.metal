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


inline uint swizzled_column_index(uint column, uint inner) {
    return (column & ~7u) | ((column & 7u) ^ (inner & 7u));
}

kernel void packed_swizzled_b_gemm_4x4(
    device const float *a [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
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
    uint kTileCount = (uniforms.k + blockK - 1) / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint tileStart = tileIndex * blockK;
        for (uint loadIndex = localLinearIndex; loadIndex < blockM * blockK; loadIndex += 64) {
            uint rowIndex = loadIndex / blockK;
            uint innerIndex = loadIndex % blockK;
            uint globalACol = tileStart + innerIndex;
            uint globalARow = threadgroupPosition.y * blockM + rowIndex;
            tileA[rowIndex][innerIndex] =
                (globalARow < uniforms.m && globalACol < uniforms.k) ? a[globalARow * uniforms.k + globalACol] : 0.0f;
        }

        uint packedTileBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * blockN;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * blockN; loadIndex += 64) {
            uint innerIndex = loadIndex / blockN;
            uint swizzledColumn = loadIndex % blockN;
            tileB[innerIndex][swizzledColumn] = packedB[packedTileBase + innerIndex * blockN + swizzledColumn];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < blockK; inner++) {
            float4 bVector = float4(
                tileB[inner][swizzled_column_index(localColBase + 0, inner)],
                tileB[inner][swizzled_column_index(localColBase + 1, inner)],
                tileB[inner][swizzled_column_index(localColBase + 2, inner)],
                tileB[inner][swizzled_column_index(localColBase + 3, inner)]
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


kernel void packed_vectorized_b_gemm_4x4(
    device const float *a [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
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
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float tileA[blockM][blockK];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localRowBase = threadPositionInThreadgroup.y * registerBlockM;
    uint localColBase = threadPositionInThreadgroup.x * registerBlockN;
    uint globalRowBase = threadgroupPosition.y * blockM + localRowBase;
    uint globalColBase = threadgroupPosition.x * blockN + localColBase;
    uint kTileCount = (uniforms.k + blockK - 1) / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        for (uint loadIndex = localLinearIndex; loadIndex < blockM * blockK; loadIndex += 64) {
            uint rowIndex = loadIndex / blockK;
            uint innerIndex = loadIndex % blockK;
            uint globalACol = tileIndex * blockK + innerIndex;
            uint globalARow = threadgroupPosition.y * blockM + rowIndex;
            tileA[rowIndex][innerIndex] =
                (globalARow < uniforms.m && globalACol < uniforms.k) ? a[globalARow * uniforms.k + globalACol] : 0.0f;
        }

        uint packedVectorBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        uint innerLoadIndex = threadPositionInThreadgroup.y;
        uint vectorLoadIndex = threadPositionInThreadgroup.x;
        tileB[innerLoadIndex][vectorLoadIndex] = float4(
            packedBVectors[packedVectorBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
        );

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < blockK; inner++) {
            float4 bVector = tileB[inner][threadPositionInThreadgroup.x];
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


inline uint swizzled_vector_index(uint vectorIndex, uint inner) {
    return (vectorIndex & ~3u) | ((vectorIndex & 3u) ^ (inner & 3u));
}

kernel void packed_swizzled_vectorized_b_gemm_4x4(
    device const float *a [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
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
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float tileA[blockM][blockK];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localRowBase = threadPositionInThreadgroup.y * registerBlockM;
    uint localColBase = threadPositionInThreadgroup.x * registerBlockN;
    uint globalRowBase = threadgroupPosition.y * blockM + localRowBase;
    uint globalColBase = threadgroupPosition.x * blockN + localColBase;
    uint kTileCount = (uniforms.k + blockK - 1) / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        for (uint loadIndex = localLinearIndex; loadIndex < blockM * blockK; loadIndex += 64) {
            uint rowIndex = loadIndex / blockK;
            uint innerIndex = loadIndex % blockK;
            uint globalACol = tileIndex * blockK + innerIndex;
            uint globalARow = threadgroupPosition.y * blockM + rowIndex;
            tileA[rowIndex][innerIndex] =
                (globalARow < uniforms.m && globalACol < uniforms.k) ? a[globalARow * uniforms.k + globalACol] : 0.0f;
        }

        uint packedVectorBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        uint innerLoadIndex = threadPositionInThreadgroup.y;
        uint vectorLoadIndex = threadPositionInThreadgroup.x;
        tileB[innerLoadIndex][vectorLoadIndex] = float4(
            packedBVectors[packedVectorBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
        );

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < blockK; inner++) {
            float4 bVector = tileB[inner][swizzled_vector_index(threadPositionInThreadgroup.x, inner)];
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

kernel void packed_vectorized_b_gemm_4x4_k16(
    device const float *a [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 32;
    constexpr uint blockN = 32;
    constexpr uint blockK = 16;
    constexpr uint registerBlockM = 4;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 8;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float tileA[blockM][blockK];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localRowBase = threadPositionInThreadgroup.y * registerBlockM;
    uint globalRowBase = threadgroupPosition.y * blockM + localRowBase;
    uint globalColBase = threadgroupPosition.x * blockN + threadPositionInThreadgroup.x * registerBlockN;
    uint kTileCount = (uniforms.k + blockK - 1) / blockK;
    bool alignedProblem = (uniforms.m % blockM == 0) && (uniforms.n % blockN == 0) && (uniforms.k % blockK == 0);

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint tileStart = tileIndex * blockK;
        for (uint loadIndex = localLinearIndex; loadIndex < blockM * blockK; loadIndex += 64) {
            uint rowIndex = loadIndex / blockK;
            uint innerIndex = loadIndex % blockK;
            uint globalARow = threadgroupPosition.y * blockM + rowIndex;
            uint globalACol = tileStart + innerIndex;
            if (alignedProblem) {
                tileA[rowIndex][innerIndex] = a[globalARow * uniforms.k + globalACol];
            } else {
                tileA[rowIndex][innerIndex] =
                    (globalARow < uniforms.m && globalACol < uniforms.k) ? a[globalARow * uniforms.k + globalACol] : 0.0f;
            }
        }

        uint packedVectorBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += 64) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedVectorBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < blockK; inner++) {
            float4 bVector = tileB[inner][threadPositionInThreadgroup.x];
            accumulator0 += tileA[localRowBase + 0][inner] * bVector;
            accumulator1 += tileA[localRowBase + 1][inner] * bVector;
            accumulator2 += tileA[localRowBase + 2][inner] * bVector;
            accumulator3 += tileA[localRowBase + 3][inner] * bVector;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (alignedProblem) {
        c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
        c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
        c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
        c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
        c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
        c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
        c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
        c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
        c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
        c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
        c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
        c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
        c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
        c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
        c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
        c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
        return;
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

kernel void packed_vectorized_a_b_gemm_4x4_k16(
    device const float *packedA [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 32;
    constexpr uint blockN = 32;
    constexpr uint blockK = 16;
    constexpr uint registerBlockM = 4;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 8;
    constexpr uint aVectorsPerInner = blockM / registerBlockM;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float4 tileA[blockK][aVectorsPerInner];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint globalRowBase = threadgroupPosition.y * blockM + threadPositionInThreadgroup.y * registerBlockM;
    uint globalColBase = threadgroupPosition.x * blockN + threadPositionInThreadgroup.x * registerBlockN;
    uint kTileCount = (uniforms.k + blockK - 1) / blockK;
    bool alignedProblem = (uniforms.m % blockM == 0) && (uniforms.n % blockN == 0) && (uniforms.k % blockK == 0);

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedAVectors = reinterpret_cast<const device packed_float4 *>(packedA);
    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint packedABase = (threadgroupPosition.y * kTileCount + tileIndex) * blockK * aVectorsPerInner;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += 64) {
            uint innerLoadIndex = loadIndex / aVectorsPerInner;
            uint vectorLoadIndex = loadIndex % aVectorsPerInner;
            tileA[innerLoadIndex][vectorLoadIndex] = float4(
                packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
            );
        }

        uint packedBBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += 64) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < blockK; inner++) {
            float4 aVector = tileA[inner][threadPositionInThreadgroup.y];
            float4 bVector = tileB[inner][threadPositionInThreadgroup.x];
            accumulator0 += aVector[0] * bVector;
            accumulator1 += aVector[1] * bVector;
            accumulator2 += aVector[2] * bVector;
            accumulator3 += aVector[3] * bVector;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (alignedProblem) {
        c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
        c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
        c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
        c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
        c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
        c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
        c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
        c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
        c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
        c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
        c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
        c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
        c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
        c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
        c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
        c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
        return;
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

inline void accumulate_register_block_4x4(
    float4 aVector,
    float4 bVector,
    thread float4 &accumulator0,
    thread float4 &accumulator1,
    thread float4 &accumulator2,
    thread float4 &accumulator3
) {
    accumulator0 += aVector[0] * bVector;
    accumulator1 += aVector[1] * bVector;
    accumulator2 += aVector[2] * bVector;
    accumulator3 += aVector[3] * bVector;
}

kernel void packed_vectorized_a_b_gemm_4x4_k16_unrolled(
    device const float *packedA [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 32;
    constexpr uint blockN = 32;
    constexpr uint blockK = 16;
    constexpr uint registerBlockM = 4;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 8;
    constexpr uint aVectorsPerInner = blockM / registerBlockM;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float4 tileA[blockK][aVectorsPerInner];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localARowVector = threadPositionInThreadgroup.y;
    uint localBVectorIndex = threadPositionInThreadgroup.x;
    uint globalRowBase = threadgroupPosition.y * blockM + threadPositionInThreadgroup.y * registerBlockM;
    uint globalColBase = threadgroupPosition.x * blockN + threadPositionInThreadgroup.x * registerBlockN;
    uint kTileCount = (uniforms.k + blockK - 1) / blockK;
    bool alignedProblem = (uniforms.m % blockM == 0) && (uniforms.n % blockN == 0) && (uniforms.k % blockK == 0);

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedAVectors = reinterpret_cast<const device packed_float4 *>(packedA);
    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint packedABase = (threadgroupPosition.y * kTileCount + tileIndex) * blockK * aVectorsPerInner;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += 64) {
            uint innerLoadIndex = loadIndex / aVectorsPerInner;
            uint vectorLoadIndex = loadIndex % aVectorsPerInner;
            tileA[innerLoadIndex][vectorLoadIndex] = float4(
                packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
            );
        }

        uint packedBBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += 64) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            float4 aVector0 = tileA[0][localARowVector];
            float4 bVector0 = tileB[0][localBVectorIndex];
            float4 aVector1 = tileA[1][localARowVector];
            float4 bVector1 = tileB[1][localBVectorIndex];
            float4 aVector2 = tileA[2][localARowVector];
            float4 bVector2 = tileB[2][localBVectorIndex];
            float4 aVector3 = tileA[3][localARowVector];
            float4 bVector3 = tileB[3][localBVectorIndex];
            accumulate_register_block_4x4(aVector0, bVector0, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector1, bVector1, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector2, bVector2, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector3, bVector3, accumulator0, accumulator1, accumulator2, accumulator3);
        }
        {
            float4 aVector4 = tileA[4][localARowVector];
            float4 bVector4 = tileB[4][localBVectorIndex];
            float4 aVector5 = tileA[5][localARowVector];
            float4 bVector5 = tileB[5][localBVectorIndex];
            float4 aVector6 = tileA[6][localARowVector];
            float4 bVector6 = tileB[6][localBVectorIndex];
            float4 aVector7 = tileA[7][localARowVector];
            float4 bVector7 = tileB[7][localBVectorIndex];
            accumulate_register_block_4x4(aVector4, bVector4, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector5, bVector5, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector6, bVector6, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector7, bVector7, accumulator0, accumulator1, accumulator2, accumulator3);
        }
        {
            float4 aVector8 = tileA[8][localARowVector];
            float4 bVector8 = tileB[8][localBVectorIndex];
            float4 aVector9 = tileA[9][localARowVector];
            float4 bVector9 = tileB[9][localBVectorIndex];
            float4 aVector10 = tileA[10][localARowVector];
            float4 bVector10 = tileB[10][localBVectorIndex];
            float4 aVector11 = tileA[11][localARowVector];
            float4 bVector11 = tileB[11][localBVectorIndex];
            accumulate_register_block_4x4(aVector8, bVector8, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector9, bVector9, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector10, bVector10, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector11, bVector11, accumulator0, accumulator1, accumulator2, accumulator3);
        }
        {
            float4 aVector12 = tileA[12][localARowVector];
            float4 bVector12 = tileB[12][localBVectorIndex];
            float4 aVector13 = tileA[13][localARowVector];
            float4 bVector13 = tileB[13][localBVectorIndex];
            float4 aVector14 = tileA[14][localARowVector];
            float4 bVector14 = tileB[14][localBVectorIndex];
            float4 aVector15 = tileA[15][localARowVector];
            float4 bVector15 = tileB[15][localBVectorIndex];
            accumulate_register_block_4x4(aVector12, bVector12, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector13, bVector13, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector14, bVector14, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector15, bVector15, accumulator0, accumulator1, accumulator2, accumulator3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (alignedProblem) {
        c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
        c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
        c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
        c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
        c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
        c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
        c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
        c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
        c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
        c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
        c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
        c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
        c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
        c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
        c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
        c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
        return;
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

kernel void packed_vectorized_a_b_gemm_4x4_k16_aligned(
    device const float *packedA [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 32;
    constexpr uint blockN = 32;
    constexpr uint blockK = 16;
    constexpr uint registerBlockM = 4;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 8;
    constexpr uint aVectorsPerInner = blockM / registerBlockM;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float4 tileA[blockK][aVectorsPerInner];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localARowVector = threadPositionInThreadgroup.y;
    uint localBVectorIndex = threadPositionInThreadgroup.x;
    uint globalRowBase = threadgroupPosition.y * blockM + localARowVector * registerBlockM;
    uint globalColBase = threadgroupPosition.x * blockN + localBVectorIndex * registerBlockN;
    uint kTileCount = uniforms.k / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedAVectors = reinterpret_cast<const device packed_float4 *>(packedA);
    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint packedABase = (threadgroupPosition.y * kTileCount + tileIndex) * blockK * aVectorsPerInner;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += 64) {
            uint innerLoadIndex = loadIndex / aVectorsPerInner;
            uint vectorLoadIndex = loadIndex % aVectorsPerInner;
            tileA[innerLoadIndex][vectorLoadIndex] = float4(
                packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
            );
        }

        uint packedBBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += 64) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < blockK; inner++) {
            float4 aVector = tileA[inner][localARowVector];
            float4 bVector = tileB[inner][localBVectorIndex];
            accumulator0 += aVector[0] * bVector;
            accumulator1 += aVector[1] * bVector;
            accumulator2 += aVector[2] * bVector;
            accumulator3 += aVector[3] * bVector;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
}

kernel void packed_vectorized_a_b_gemm_4x4_64x32x16_aligned(
    device const float *packedA [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 64;
    constexpr uint blockN = 32;
    constexpr uint blockK = 16;
    constexpr uint registerBlockM = 4;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 8;
    constexpr uint threadsPerThreadgroupY = 16;
    constexpr uint threadsPerThreadgroup = threadsPerThreadgroupX * threadsPerThreadgroupY;
    constexpr uint aVectorsPerInner = blockM / registerBlockM;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float4 tileA[blockK][aVectorsPerInner];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localARowVector = threadPositionInThreadgroup.y;
    uint localBVectorIndex = threadPositionInThreadgroup.x;
    uint globalRowBase = threadgroupPosition.y * blockM + localARowVector * registerBlockM;
    uint globalColBase = threadgroupPosition.x * blockN + localBVectorIndex * registerBlockN;
    uint kTileCount = uniforms.k / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedAVectors = reinterpret_cast<const device packed_float4 *>(packedA);
    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint packedABase = (threadgroupPosition.y * kTileCount + tileIndex) * blockK * aVectorsPerInner;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / aVectorsPerInner;
            uint vectorLoadIndex = loadIndex % aVectorsPerInner;
            tileA[innerLoadIndex][vectorLoadIndex] = float4(
                packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
            );
        }

        uint packedBBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < blockK; inner++) {
            float4 aVector = tileA[inner][localARowVector];
            float4 bVector = tileB[inner][localBVectorIndex];
            accumulate_register_block_4x4(aVector, bVector, accumulator0, accumulator1, accumulator2, accumulator3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
}

kernel void packed_vectorized_a_b_gemm_4x4_64x32x16_unrolled(
    device const float *packedA [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 64;
    constexpr uint blockN = 32;
    constexpr uint blockK = 16;
    constexpr uint registerBlockM = 4;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 8;
    constexpr uint threadsPerThreadgroupY = 16;
    constexpr uint threadsPerThreadgroup = threadsPerThreadgroupX * threadsPerThreadgroupY;
    constexpr uint aVectorsPerInner = blockM / registerBlockM;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float4 tileA[blockK][aVectorsPerInner];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localARowVector = threadPositionInThreadgroup.y;
    uint localBVectorIndex = threadPositionInThreadgroup.x;
    uint globalRowBase = threadgroupPosition.y * blockM + localARowVector * registerBlockM;
    uint globalColBase = threadgroupPosition.x * blockN + localBVectorIndex * registerBlockN;
    uint kTileCount = uniforms.k / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedAVectors = reinterpret_cast<const device packed_float4 *>(packedA);
    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint packedABase = (threadgroupPosition.y * kTileCount + tileIndex) * blockK * aVectorsPerInner;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / aVectorsPerInner;
            uint vectorLoadIndex = loadIndex % aVectorsPerInner;
            tileA[innerLoadIndex][vectorLoadIndex] = float4(
                packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
            );
        }

        uint packedBBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            float4 aVector0 = tileA[0][localARowVector];
            float4 bVector0 = tileB[0][localBVectorIndex];
            float4 aVector1 = tileA[1][localARowVector];
            float4 bVector1 = tileB[1][localBVectorIndex];
            float4 aVector2 = tileA[2][localARowVector];
            float4 bVector2 = tileB[2][localBVectorIndex];
            float4 aVector3 = tileA[3][localARowVector];
            float4 bVector3 = tileB[3][localBVectorIndex];
            accumulate_register_block_4x4(aVector0, bVector0, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector1, bVector1, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector2, bVector2, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector3, bVector3, accumulator0, accumulator1, accumulator2, accumulator3);
        }
        {
            float4 aVector4 = tileA[4][localARowVector];
            float4 bVector4 = tileB[4][localBVectorIndex];
            float4 aVector5 = tileA[5][localARowVector];
            float4 bVector5 = tileB[5][localBVectorIndex];
            float4 aVector6 = tileA[6][localARowVector];
            float4 bVector6 = tileB[6][localBVectorIndex];
            float4 aVector7 = tileA[7][localARowVector];
            float4 bVector7 = tileB[7][localBVectorIndex];
            accumulate_register_block_4x4(aVector4, bVector4, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector5, bVector5, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector6, bVector6, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector7, bVector7, accumulator0, accumulator1, accumulator2, accumulator3);
        }
        {
            float4 aVector8 = tileA[8][localARowVector];
            float4 bVector8 = tileB[8][localBVectorIndex];
            float4 aVector9 = tileA[9][localARowVector];
            float4 bVector9 = tileB[9][localBVectorIndex];
            float4 aVector10 = tileA[10][localARowVector];
            float4 bVector10 = tileB[10][localBVectorIndex];
            float4 aVector11 = tileA[11][localARowVector];
            float4 bVector11 = tileB[11][localBVectorIndex];
            accumulate_register_block_4x4(aVector8, bVector8, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector9, bVector9, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector10, bVector10, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector11, bVector11, accumulator0, accumulator1, accumulator2, accumulator3);
        }
        {
            float4 aVector12 = tileA[12][localARowVector];
            float4 bVector12 = tileB[12][localBVectorIndex];
            float4 aVector13 = tileA[13][localARowVector];
            float4 bVector13 = tileB[13][localBVectorIndex];
            float4 aVector14 = tileA[14][localARowVector];
            float4 bVector14 = tileB[14][localBVectorIndex];
            float4 aVector15 = tileA[15][localARowVector];
            float4 bVector15 = tileB[15][localBVectorIndex];
            accumulate_register_block_4x4(aVector12, bVector12, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector13, bVector13, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector14, bVector14, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector15, bVector15, accumulator0, accumulator1, accumulator2, accumulator3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
}

kernel void packed_vectorized_a_b_gemm_4x4_64x32x16_pipelined(
    device const float *packedA [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 64;
    constexpr uint blockN = 32;
    constexpr uint blockK = 16;
    constexpr uint registerBlockM = 4;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 8;
    constexpr uint threadsPerThreadgroupY = 16;
    constexpr uint threadsPerThreadgroup = threadsPerThreadgroupX * threadsPerThreadgroupY;
    constexpr uint aVectorsPerInner = blockM / registerBlockM;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float4 tileA[2][blockK][aVectorsPerInner];
    threadgroup float4 tileB[2][blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localARowVector = threadPositionInThreadgroup.y;
    uint localBVectorIndex = threadPositionInThreadgroup.x;
    uint globalRowBase = threadgroupPosition.y * blockM + localARowVector * registerBlockM;
    uint globalColBase = threadgroupPosition.x * blockN + localBVectorIndex * registerBlockN;
    uint kTileCount = uniforms.k / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedAVectors = reinterpret_cast<const device packed_float4 *>(packedA);
    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    {
        uint packedABase = threadgroupPosition.y * kTileCount * blockK * aVectorsPerInner;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / aVectorsPerInner;
            uint vectorLoadIndex = loadIndex % aVectorsPerInner;
            tileA[0][innerLoadIndex][vectorLoadIndex] = float4(
                packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
            );
        }

        uint packedBBase = threadgroupPosition.x * kTileCount * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[0][innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint stage = tileIndex & 1u;
        uint nextStage = stage ^ 1u;

        for (uint inner = 0; inner < blockK; inner++) {
            float4 aVector = tileA[stage][inner][localARowVector];
            float4 bVector = tileB[stage][inner][localBVectorIndex];
            accumulate_register_block_4x4(aVector, bVector, accumulator0, accumulator1, accumulator2, accumulator3);
        }

        if (tileIndex + 1 < kTileCount) {
            uint nextTileIndex = tileIndex + 1;
            uint packedABase = (threadgroupPosition.y * kTileCount + nextTileIndex) * blockK * aVectorsPerInner;
            for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += threadsPerThreadgroup) {
                uint innerLoadIndex = loadIndex / aVectorsPerInner;
                uint vectorLoadIndex = loadIndex % aVectorsPerInner;
                tileA[nextStage][innerLoadIndex][vectorLoadIndex] = float4(
                    packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
                );
            }

            uint packedBBase = (threadgroupPosition.x * kTileCount + nextTileIndex) * blockK * bVectorsPerRow;
            for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += threadsPerThreadgroup) {
                uint innerLoadIndex = loadIndex / bVectorsPerRow;
                uint vectorLoadIndex = loadIndex % bVectorsPerRow;
                tileB[nextStage][innerLoadIndex][vectorLoadIndex] = float4(
                    packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
                );
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
}

kernel void packed_vectorized_a_b_gemm_4x4_64x64x16_aligned(
    device const float *packedA [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 64;
    constexpr uint blockN = 64;
    constexpr uint blockK = 16;
    constexpr uint registerBlockM = 4;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 16;
    constexpr uint threadsPerThreadgroupY = 16;
    constexpr uint threadsPerThreadgroup = threadsPerThreadgroupX * threadsPerThreadgroupY;
    constexpr uint aVectorsPerInner = blockM / registerBlockM;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float4 tileA[blockK][aVectorsPerInner];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localARowVector = threadPositionInThreadgroup.y;
    uint localBVectorIndex = threadPositionInThreadgroup.x;
    uint globalRowBase = threadgroupPosition.y * blockM + localARowVector * registerBlockM;
    uint globalColBase = threadgroupPosition.x * blockN + localBVectorIndex * registerBlockN;
    uint kTileCount = uniforms.k / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedAVectors = reinterpret_cast<const device packed_float4 *>(packedA);
    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint packedABase = (threadgroupPosition.y * kTileCount + tileIndex) * blockK * aVectorsPerInner;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / aVectorsPerInner;
            uint vectorLoadIndex = loadIndex % aVectorsPerInner;
            tileA[innerLoadIndex][vectorLoadIndex] = float4(
                packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
            );
        }

        uint packedBBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < blockK; inner++) {
            float4 aVector = tileA[inner][localARowVector];
            float4 bVector = tileB[inner][localBVectorIndex];
            accumulate_register_block_4x4(aVector, bVector, accumulator0, accumulator1, accumulator2, accumulator3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
}

kernel void packed_vectorized_a_b_gemm_4x4_64x64x16_unrolled(
    device const float *packedA [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 64;
    constexpr uint blockN = 64;
    constexpr uint blockK = 16;
    constexpr uint registerBlockM = 4;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 16;
    constexpr uint threadsPerThreadgroupY = 16;
    constexpr uint threadsPerThreadgroup = threadsPerThreadgroupX * threadsPerThreadgroupY;
    constexpr uint aVectorsPerInner = blockM / registerBlockM;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float4 tileA[blockK][aVectorsPerInner];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localARowVector = threadPositionInThreadgroup.y;
    uint localBVectorIndex = threadPositionInThreadgroup.x;
    uint globalRowBase = threadgroupPosition.y * blockM + localARowVector * registerBlockM;
    uint globalColBase = threadgroupPosition.x * blockN + localBVectorIndex * registerBlockN;
    uint kTileCount = uniforms.k / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedAVectors = reinterpret_cast<const device packed_float4 *>(packedA);
    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint packedABase = (threadgroupPosition.y * kTileCount + tileIndex) * blockK * aVectorsPerInner;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / aVectorsPerInner;
            uint vectorLoadIndex = loadIndex % aVectorsPerInner;
            tileA[innerLoadIndex][vectorLoadIndex] = float4(
                packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
            );
        }

        uint packedBBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            float4 aVector0 = tileA[0][localARowVector];
            float4 bVector0 = tileB[0][localBVectorIndex];
            float4 aVector1 = tileA[1][localARowVector];
            float4 bVector1 = tileB[1][localBVectorIndex];
            float4 aVector2 = tileA[2][localARowVector];
            float4 bVector2 = tileB[2][localBVectorIndex];
            float4 aVector3 = tileA[3][localARowVector];
            float4 bVector3 = tileB[3][localBVectorIndex];
            accumulate_register_block_4x4(aVector0, bVector0, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector1, bVector1, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector2, bVector2, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector3, bVector3, accumulator0, accumulator1, accumulator2, accumulator3);
        }
        {
            float4 aVector4 = tileA[4][localARowVector];
            float4 bVector4 = tileB[4][localBVectorIndex];
            float4 aVector5 = tileA[5][localARowVector];
            float4 bVector5 = tileB[5][localBVectorIndex];
            float4 aVector6 = tileA[6][localARowVector];
            float4 bVector6 = tileB[6][localBVectorIndex];
            float4 aVector7 = tileA[7][localARowVector];
            float4 bVector7 = tileB[7][localBVectorIndex];
            accumulate_register_block_4x4(aVector4, bVector4, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector5, bVector5, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector6, bVector6, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector7, bVector7, accumulator0, accumulator1, accumulator2, accumulator3);
        }
        {
            float4 aVector8 = tileA[8][localARowVector];
            float4 bVector8 = tileB[8][localBVectorIndex];
            float4 aVector9 = tileA[9][localARowVector];
            float4 bVector9 = tileB[9][localBVectorIndex];
            float4 aVector10 = tileA[10][localARowVector];
            float4 bVector10 = tileB[10][localBVectorIndex];
            float4 aVector11 = tileA[11][localARowVector];
            float4 bVector11 = tileB[11][localBVectorIndex];
            accumulate_register_block_4x4(aVector8, bVector8, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector9, bVector9, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector10, bVector10, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector11, bVector11, accumulator0, accumulator1, accumulator2, accumulator3);
        }
        {
            float4 aVector12 = tileA[12][localARowVector];
            float4 bVector12 = tileB[12][localBVectorIndex];
            float4 aVector13 = tileA[13][localARowVector];
            float4 bVector13 = tileB[13][localBVectorIndex];
            float4 aVector14 = tileA[14][localARowVector];
            float4 bVector14 = tileB[14][localBVectorIndex];
            float4 aVector15 = tileA[15][localARowVector];
            float4 bVector15 = tileB[15][localBVectorIndex];
            accumulate_register_block_4x4(aVector12, bVector12, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector13, bVector13, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector14, bVector14, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector15, bVector15, accumulator0, accumulator1, accumulator2, accumulator3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
}

kernel void packed_vectorized_a_b_gemm_4x4_64x64x32_aligned(
    device const float *packedA [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 64;
    constexpr uint blockN = 64;
    constexpr uint blockK = 32;
    constexpr uint registerBlockM = 4;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 16;
    constexpr uint threadsPerThreadgroupY = 16;
    constexpr uint threadsPerThreadgroup = threadsPerThreadgroupX * threadsPerThreadgroupY;
    constexpr uint aVectorsPerInner = blockM / registerBlockM;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float4 tileA[blockK][aVectorsPerInner];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localARowVector = threadPositionInThreadgroup.y;
    uint localBVectorIndex = threadPositionInThreadgroup.x;
    uint globalRowBase = threadgroupPosition.y * blockM + localARowVector * registerBlockM;
    uint globalColBase = threadgroupPosition.x * blockN + localBVectorIndex * registerBlockN;
    uint kTileCount = uniforms.k / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedAVectors = reinterpret_cast<const device packed_float4 *>(packedA);
    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint packedABase = (threadgroupPosition.y * kTileCount + tileIndex) * blockK * aVectorsPerInner;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / aVectorsPerInner;
            uint vectorLoadIndex = loadIndex % aVectorsPerInner;
            tileA[innerLoadIndex][vectorLoadIndex] = float4(
                packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
            );
        }

        uint packedBBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < blockK; inner++) {
            float4 aVector = tileA[inner][localARowVector];
            float4 bVector = tileB[inner][localBVectorIndex];
            accumulate_register_block_4x4(aVector, bVector, accumulator0, accumulator1, accumulator2, accumulator3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
}

kernel void packed_vectorized_a_b_gemm_8x4_64x64x16_aligned(
    device const float *packedA [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 64;
    constexpr uint blockN = 64;
    constexpr uint blockK = 16;
    constexpr uint registerBlockM = 8;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 16;
    constexpr uint threadsPerThreadgroupY = 8;
    constexpr uint threadsPerThreadgroup = threadsPerThreadgroupX * threadsPerThreadgroupY;
    constexpr uint aVectorsPerInner = blockM / 4;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float4 tileA[blockK][aVectorsPerInner];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localARowBlock = threadPositionInThreadgroup.y;
    uint localAVectorBase = localARowBlock * 2;
    uint localBVectorIndex = threadPositionInThreadgroup.x;
    uint globalRowBase = threadgroupPosition.y * blockM + localARowBlock * registerBlockM;
    uint globalColBase = threadgroupPosition.x * blockN + localBVectorIndex * registerBlockN;
    uint kTileCount = uniforms.k / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);
    float4 accumulator4 = float4(0.0f);
    float4 accumulator5 = float4(0.0f);
    float4 accumulator6 = float4(0.0f);
    float4 accumulator7 = float4(0.0f);

    const device packed_float4 *packedAVectors = reinterpret_cast<const device packed_float4 *>(packedA);
    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint packedABase = (threadgroupPosition.y * kTileCount + tileIndex) * blockK * aVectorsPerInner;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / aVectorsPerInner;
            uint vectorLoadIndex = loadIndex % aVectorsPerInner;
            tileA[innerLoadIndex][vectorLoadIndex] = float4(
                packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
            );
        }

        uint packedBBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < blockK; inner++) {
            float4 aVector0 = tileA[inner][localAVectorBase + 0];
            float4 aVector1 = tileA[inner][localAVectorBase + 1];
            float4 bVector = tileB[inner][localBVectorIndex];
            accumulate_register_block_4x4(aVector0, bVector, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector1, bVector, accumulator4, accumulator5, accumulator6, accumulator7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
    c[(globalRowBase + 4) * uniforms.n + globalColBase + 0] = accumulator4[0];
    c[(globalRowBase + 4) * uniforms.n + globalColBase + 1] = accumulator4[1];
    c[(globalRowBase + 4) * uniforms.n + globalColBase + 2] = accumulator4[2];
    c[(globalRowBase + 4) * uniforms.n + globalColBase + 3] = accumulator4[3];
    c[(globalRowBase + 5) * uniforms.n + globalColBase + 0] = accumulator5[0];
    c[(globalRowBase + 5) * uniforms.n + globalColBase + 1] = accumulator5[1];
    c[(globalRowBase + 5) * uniforms.n + globalColBase + 2] = accumulator5[2];
    c[(globalRowBase + 5) * uniforms.n + globalColBase + 3] = accumulator5[3];
    c[(globalRowBase + 6) * uniforms.n + globalColBase + 0] = accumulator6[0];
    c[(globalRowBase + 6) * uniforms.n + globalColBase + 1] = accumulator6[1];
    c[(globalRowBase + 6) * uniforms.n + globalColBase + 2] = accumulator6[2];
    c[(globalRowBase + 6) * uniforms.n + globalColBase + 3] = accumulator6[3];
    c[(globalRowBase + 7) * uniforms.n + globalColBase + 0] = accumulator7[0];
    c[(globalRowBase + 7) * uniforms.n + globalColBase + 1] = accumulator7[1];
    c[(globalRowBase + 7) * uniforms.n + globalColBase + 2] = accumulator7[2];
    c[(globalRowBase + 7) * uniforms.n + globalColBase + 3] = accumulator7[3];
}

kernel void packed_vectorized_a_b_gemm_8x4_64x128x16_aligned(
    device const float *packedA [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 64;
    constexpr uint blockN = 128;
    constexpr uint blockK = 16;
    constexpr uint registerBlockM = 8;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 32;
    constexpr uint threadsPerThreadgroupY = 8;
    constexpr uint threadsPerThreadgroup = threadsPerThreadgroupX * threadsPerThreadgroupY;
    constexpr uint aVectorsPerInner = blockM / 4;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float4 tileA[blockK][aVectorsPerInner];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localARowBlock = threadPositionInThreadgroup.y;
    uint localAVectorBase = localARowBlock * 2;
    uint localBVectorIndex = threadPositionInThreadgroup.x;
    uint globalRowBase = threadgroupPosition.y * blockM + localARowBlock * registerBlockM;
    uint globalColBase = threadgroupPosition.x * blockN + localBVectorIndex * registerBlockN;
    uint kTileCount = uniforms.k / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);
    float4 accumulator4 = float4(0.0f);
    float4 accumulator5 = float4(0.0f);
    float4 accumulator6 = float4(0.0f);
    float4 accumulator7 = float4(0.0f);

    const device packed_float4 *packedAVectors = reinterpret_cast<const device packed_float4 *>(packedA);
    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint packedABase = (threadgroupPosition.y * kTileCount + tileIndex) * blockK * aVectorsPerInner;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / aVectorsPerInner;
            uint vectorLoadIndex = loadIndex % aVectorsPerInner;
            tileA[innerLoadIndex][vectorLoadIndex] = float4(
                packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
            );
        }

        uint packedBBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < blockK; inner++) {
            float4 aVector0 = tileA[inner][localAVectorBase + 0];
            float4 aVector1 = tileA[inner][localAVectorBase + 1];
            float4 bVector = tileB[inner][localBVectorIndex];
            accumulate_register_block_4x4(aVector0, bVector, accumulator0, accumulator1, accumulator2, accumulator3);
            accumulate_register_block_4x4(aVector1, bVector, accumulator4, accumulator5, accumulator6, accumulator7);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
    c[(globalRowBase + 4) * uniforms.n + globalColBase + 0] = accumulator4[0];
    c[(globalRowBase + 4) * uniforms.n + globalColBase + 1] = accumulator4[1];
    c[(globalRowBase + 4) * uniforms.n + globalColBase + 2] = accumulator4[2];
    c[(globalRowBase + 4) * uniforms.n + globalColBase + 3] = accumulator4[3];
    c[(globalRowBase + 5) * uniforms.n + globalColBase + 0] = accumulator5[0];
    c[(globalRowBase + 5) * uniforms.n + globalColBase + 1] = accumulator5[1];
    c[(globalRowBase + 5) * uniforms.n + globalColBase + 2] = accumulator5[2];
    c[(globalRowBase + 5) * uniforms.n + globalColBase + 3] = accumulator5[3];
    c[(globalRowBase + 6) * uniforms.n + globalColBase + 0] = accumulator6[0];
    c[(globalRowBase + 6) * uniforms.n + globalColBase + 1] = accumulator6[1];
    c[(globalRowBase + 6) * uniforms.n + globalColBase + 2] = accumulator6[2];
    c[(globalRowBase + 6) * uniforms.n + globalColBase + 3] = accumulator6[3];
    c[(globalRowBase + 7) * uniforms.n + globalColBase + 0] = accumulator7[0];
    c[(globalRowBase + 7) * uniforms.n + globalColBase + 1] = accumulator7[1];
    c[(globalRowBase + 7) * uniforms.n + globalColBase + 2] = accumulator7[2];
    c[(globalRowBase + 7) * uniforms.n + globalColBase + 3] = accumulator7[3];
}

kernel void packed_vectorized_a_b_gemm_4x4_32x64x16_aligned(
    device const float *packedA [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 32;
    constexpr uint blockN = 64;
    constexpr uint blockK = 16;
    constexpr uint registerBlockM = 4;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 16;
    constexpr uint threadsPerThreadgroupY = 8;
    constexpr uint threadsPerThreadgroup = threadsPerThreadgroupX * threadsPerThreadgroupY;
    constexpr uint aVectorsPerInner = blockM / registerBlockM;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float4 tileA[blockK][aVectorsPerInner];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localARowVector = threadPositionInThreadgroup.y;
    uint localBVectorIndex = threadPositionInThreadgroup.x;
    uint globalRowBase = threadgroupPosition.y * blockM + localARowVector * registerBlockM;
    uint globalColBase = threadgroupPosition.x * blockN + localBVectorIndex * registerBlockN;
    uint kTileCount = uniforms.k / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedAVectors = reinterpret_cast<const device packed_float4 *>(packedA);
    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint packedABase = (threadgroupPosition.y * kTileCount + tileIndex) * blockK * aVectorsPerInner;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / aVectorsPerInner;
            uint vectorLoadIndex = loadIndex % aVectorsPerInner;
            tileA[innerLoadIndex][vectorLoadIndex] = float4(
                packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
            );
        }

        uint packedBBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < blockK; inner++) {
            float4 aVector = tileA[inner][localARowVector];
            float4 bVector = tileB[inner][localBVectorIndex];
            accumulate_register_block_4x4(aVector, bVector, accumulator0, accumulator1, accumulator2, accumulator3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
}

kernel void packed_vectorized_a_b_gemm_4x4_32x32x32_aligned(
    device const float *packedA [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 32;
    constexpr uint blockN = 32;
    constexpr uint blockK = 32;
    constexpr uint registerBlockM = 4;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 8;
    constexpr uint threadsPerThreadgroupY = 8;
    constexpr uint threadsPerThreadgroup = threadsPerThreadgroupX * threadsPerThreadgroupY;
    constexpr uint aVectorsPerInner = blockM / registerBlockM;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float4 tileA[blockK][aVectorsPerInner];
    threadgroup float4 tileB[blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localARowVector = threadPositionInThreadgroup.y;
    uint localBVectorIndex = threadPositionInThreadgroup.x;
    uint globalRowBase = threadgroupPosition.y * blockM + localARowVector * registerBlockM;
    uint globalColBase = threadgroupPosition.x * blockN + localBVectorIndex * registerBlockN;
    uint kTileCount = uniforms.k / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedAVectors = reinterpret_cast<const device packed_float4 *>(packedA);
    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint packedABase = (threadgroupPosition.y * kTileCount + tileIndex) * blockK * aVectorsPerInner;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / aVectorsPerInner;
            uint vectorLoadIndex = loadIndex % aVectorsPerInner;
            tileA[innerLoadIndex][vectorLoadIndex] = float4(
                packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
            );
        }

        uint packedBBase = (threadgroupPosition.x * kTileCount + tileIndex) * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += threadsPerThreadgroup) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < blockK; inner++) {
            float4 aVector = tileA[inner][localARowVector];
            float4 bVector = tileB[inner][localBVectorIndex];
            accumulate_register_block_4x4(aVector, bVector, accumulator0, accumulator1, accumulator2, accumulator3);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
}

kernel void packed_vectorized_a_b_gemm_4x4_k16_aligned_pipelined(
    device const float *packedA [[buffer(0)]],
    device const float *packedB [[buffer(1)]],
    device float *c [[buffer(2)]],
    constant GEMMUniforms &uniforms [[buffer(3)]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPositionInThreadgroup [[thread_position_in_threadgroup]]
) {
    constexpr uint blockM = 32;
    constexpr uint blockN = 32;
    constexpr uint blockK = 16;
    constexpr uint registerBlockM = 4;
    constexpr uint registerBlockN = 4;
    constexpr uint threadsPerThreadgroupX = 8;
    constexpr uint aVectorsPerInner = blockM / registerBlockM;
    constexpr uint bVectorsPerRow = blockN / registerBlockN;

    threadgroup float4 tileA[2][blockK][aVectorsPerInner];
    threadgroup float4 tileB[2][blockK][bVectorsPerRow];

    uint localLinearIndex = threadPositionInThreadgroup.y * threadsPerThreadgroupX + threadPositionInThreadgroup.x;
    uint localARowVector = threadPositionInThreadgroup.y;
    uint localBVectorIndex = threadPositionInThreadgroup.x;
    uint globalRowBase = threadgroupPosition.y * blockM + localARowVector * registerBlockM;
    uint globalColBase = threadgroupPosition.x * blockN + localBVectorIndex * registerBlockN;
    uint kTileCount = uniforms.k / blockK;

    float4 accumulator0 = float4(0.0f);
    float4 accumulator1 = float4(0.0f);
    float4 accumulator2 = float4(0.0f);
    float4 accumulator3 = float4(0.0f);

    const device packed_float4 *packedAVectors = reinterpret_cast<const device packed_float4 *>(packedA);
    const device packed_float4 *packedBVectors = reinterpret_cast<const device packed_float4 *>(packedB);

    {
        uint packedABase = threadgroupPosition.y * kTileCount * blockK * aVectorsPerInner;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += 64) {
            uint innerLoadIndex = loadIndex / aVectorsPerInner;
            uint vectorLoadIndex = loadIndex % aVectorsPerInner;
            tileA[0][innerLoadIndex][vectorLoadIndex] = float4(
                packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
            );
        }

        uint packedBBase = threadgroupPosition.x * kTileCount * blockK * bVectorsPerRow;
        for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += 64) {
            uint innerLoadIndex = loadIndex / bVectorsPerRow;
            uint vectorLoadIndex = loadIndex % bVectorsPerRow;
            tileB[0][innerLoadIndex][vectorLoadIndex] = float4(
                packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
            );
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tileIndex = 0; tileIndex < kTileCount; tileIndex++) {
        uint stage = tileIndex & 1u;
        uint nextStage = stage ^ 1u;

        for (uint inner = 0; inner < blockK; inner++) {
            float4 aVector = tileA[stage][inner][localARowVector];
            float4 bVector = tileB[stage][inner][localBVectorIndex];
            accumulator0 += aVector[0] * bVector;
            accumulator1 += aVector[1] * bVector;
            accumulator2 += aVector[2] * bVector;
            accumulator3 += aVector[3] * bVector;
        }

        if (tileIndex + 1 < kTileCount) {
            uint nextTileIndex = tileIndex + 1;
            uint packedABase = (threadgroupPosition.y * kTileCount + nextTileIndex) * blockK * aVectorsPerInner;
            for (uint loadIndex = localLinearIndex; loadIndex < blockK * aVectorsPerInner; loadIndex += 64) {
                uint innerLoadIndex = loadIndex / aVectorsPerInner;
                uint vectorLoadIndex = loadIndex % aVectorsPerInner;
                tileA[nextStage][innerLoadIndex][vectorLoadIndex] = float4(
                    packedAVectors[packedABase + innerLoadIndex * aVectorsPerInner + vectorLoadIndex]
                );
            }

            uint packedBBase = (threadgroupPosition.x * kTileCount + nextTileIndex) * blockK * bVectorsPerRow;
            for (uint loadIndex = localLinearIndex; loadIndex < blockK * bVectorsPerRow; loadIndex += 64) {
                uint innerLoadIndex = loadIndex / bVectorsPerRow;
                uint vectorLoadIndex = loadIndex % bVectorsPerRow;
                tileB[nextStage][innerLoadIndex][vectorLoadIndex] = float4(
                    packedBVectors[packedBBase + innerLoadIndex * bVectorsPerRow + vectorLoadIndex]
                );
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    c[(globalRowBase + 0) * uniforms.n + globalColBase + 0] = accumulator0[0];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 1] = accumulator0[1];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 2] = accumulator0[2];
    c[(globalRowBase + 0) * uniforms.n + globalColBase + 3] = accumulator0[3];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 0] = accumulator1[0];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 1] = accumulator1[1];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 2] = accumulator1[2];
    c[(globalRowBase + 1) * uniforms.n + globalColBase + 3] = accumulator1[3];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 0] = accumulator2[0];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 1] = accumulator2[1];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 2] = accumulator2[2];
    c[(globalRowBase + 2) * uniforms.n + globalColBase + 3] = accumulator2[3];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 0] = accumulator3[0];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 1] = accumulator3[1];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 2] = accumulator3[2];
    c[(globalRowBase + 3) * uniforms.n + globalColBase + 3] = accumulator3[3];
}

