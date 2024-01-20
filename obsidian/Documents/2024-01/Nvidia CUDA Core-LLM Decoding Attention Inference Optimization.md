> the calculation process of each head attention of MHA is simplified to two HGEMV and one softmax, as shown in the schematic diagram below.

![[Pasted image 20240114120846.png]]

## 3.1 HGEMV（S = Q * K^T）


```C++
    // S = Q * K^T
    half RQ[thread_elem_nums];

#pragma unroll
    for (size_t i = 0; i < thread_iters; ++i) {
        *(int4 *)(&RQ[i * thread_copy_elem_nums]) = *(int4 *)(&params.q_ptr[binfo.q_offset(
            params.q_row_stride, params.q_head_stride, (i * group_size + group_lane_id) * thread_copy_elem_nums)]);
    }

    extern __shared__ float S_smem[];
    float S_max = -std::numeric_limits<float>::max();

#pragma unroll
    for (size_t base_seqlen_k = warp_id * groups_per_warp; base_seqlen_k < binfo.actual_seqlen_k;
         base_seqlen_k += groups_per_block) {
        size_t seqlen_k = base_seqlen_k + group_id;
        half RK[thread_elem_nums];

        float tmp = 0.0;
        if (seqlen_k >= binfo.actual_seqlen_k) {
            memset(RK, 0, sizeof(RK));
        } else {
#pragma unroll
            for (size_t i = 0; i < thread_iters; ++i) {
                *(int4 *)(&RK[i * thread_copy_elem_nums]) =
                    *(int4 *)(&params.k_ptr[binfo.k_offset(seqlen_k, params.k_row_stride, params.k_head_stride,
                                                           (i * group_size + group_lane_id) * thread_copy_elem_nums)]);
            }

#pragma unroll
            for (size_t i = 0; i < thread_elem_nums; ++i) {
                tmp += (__half2float(RQ[i]) * __half2float(RK[i]));
            }
        }

#pragma unroll
        for (size_t i = group_size / 2; i >= 1; i /= 2) {
            tmp += __shfl_xor_sync(shfl_mask, tmp, i);
        }

        if (group_lane_id == 0 && seqlen_k < binfo.actual_seqlen_k) {
            tmp *= params.scale_softmax;

            if (IsAlibi) {
                tmp += (binfo.h_slope * (static_cast<int>(seqlen_k) - binfo.actual_seqlen_q - binfo.row_shift));
            }

            S_smem[seqlen_k] = tmp;
            S_max = fmaxf(tmp, S_max);
        }
    }
```

## 3.2 Softmax（P = Softmax(S)）


```C++
    // P = Softmax(S)
    __shared__ float softmax_smem[warps_per_block];

#pragma unroll
    for (size_t i = warp_size / 2; i >= 1; i /= 2) {
        S_max = fmaxf(S_max, __shfl_xor_sync(shfl_mask, S_max, i));
    }

    if (lane_id == 0) {
        softmax_smem[warp_id] = S_max;
    }

    __syncthreads();

    if (lane_id < warps_per_block) {
        S_max = softmax_smem[lane_id];
    } else {
        S_max = -std::numeric_limits<float>::max();
    }

#pragma unroll
    for (size_t i = warps_per_block / 2; i >= 1; i /= 2) {
        S_max = fmaxf(S_max, __shfl_xor_sync(shfl_mask, S_max, i));
    }

    S_max = __shfl_sync(shfl_mask, S_max, 0);

    float exp_sum = 0.0;
#pragma unroll
    for (size_t seqlen_k = threadIdx.x; seqlen_k < binfo.actual_seqlen_k; seqlen_k += threads_per_block) {
        S_smem[seqlen_k] -= S_max;
        S_smem[seqlen_k] = exp(S_smem[seqlen_k]);
        exp_sum += S_smem[seqlen_k];
    }

#pragma unroll
    for (size_t i = warp_size / 2; i >= 1; i /= 2) {
        exp_sum += __shfl_xor_sync(shfl_mask, exp_sum, i);
    }

    if (lane_id == 0) {
        softmax_smem[warp_id] = exp_sum;
    }

    __syncthreads();

    if (lane_id < warps_per_block) {
        exp_sum = softmax_smem[lane_id];
    }

#pragma unroll
    for (size_t i = warps_per_block / 2; i >= 1; i /= 2) {
        exp_sum += __shfl_xor_sync(shfl_mask, exp_sum, i);
    }
    exp_sum = __shfl_sync(shfl_mask, exp_sum, 0);

#pragma unroll
    for (size_t seqlen_k = threadIdx.x; seqlen_k < binfo.actual_seqlen_k; seqlen_k += threads_per_block) {
        S_smem[seqlen_k] /= exp_sum;
    }

    __syncthreads();
```


## HGEMM(O= P * V)

```C++
    // O = P * V
    half RV[thread_elem_nums];
    float RO[thread_elem_nums];

    memset(RO, 0, sizeof(RO));

#pragma unroll
    for (size_t base_seqlen_k = warp_id * groups_per_warp; base_seqlen_k < binfo.actual_seqlen_k;
         base_seqlen_k += groups_per_block) {
        size_t seqlen_k = base_seqlen_k + group_id;

        if (seqlen_k < binfo.actual_seqlen_k) {
#pragma unroll
            for (size_t i = 0; i < thread_iters; ++i) {
                *(int4 *)(&RV[i * thread_copy_elem_nums]) =
                    *(int4 *)(&params.v_ptr[binfo.k_offset(seqlen_k, params.v_row_stride, params.v_head_stride,
                                                           (i * group_size + group_lane_id) * thread_copy_elem_nums)]);
            }

#pragma unroll
            for (size_t i = 0; i < thread_elem_nums; ++i) {
                RO[i] += (S_smem[seqlen_k] * __half2float(RV[i]));
            }
        }
    }

#pragma unroll
    for (size_t i = 0; i < thread_elem_nums; ++i) {
#pragma unroll
        for (size_t j = group_size; j <= warp_size / 2; j *= 2) {
            RO[i] += __shfl_xor_sync(shfl_mask, RO[i], j);
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = threadIdx.x; i < head_dim; i += threads_per_block) {
        S_smem[i] = 0.0;
    }

    __syncthreads();

    if (lane_id < group_size) {
#pragma unroll
        for (size_t i = 0; i < thread_iters; ++i) {
#pragma unroll
            for (size_t j = 0; j < thread_copy_elem_nums; ++j) {
                atomicAdd(S_smem + (i * group_size + lane_id) * thread_copy_elem_nums + j,
                          RO[i * thread_copy_elem_nums + j]);
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = threadIdx.x; i < head_dim; i += threads_per_block) {
        params.o_ptr[binfo.q_offset(params.o_row_stride, params.o_head_stride, i)] = __float2half(S_smem[i]);
    }
```