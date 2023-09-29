```C++
static struct ggml_cgraph * llm_build_llama(
         llama_context & lctx,
     const llama_token * tokens,
           const float * embd,
                   int   n_tokens,
                   int   n_past) {

    GGML_ASSERT((!tokens && embd) || (tokens && !embd)); // NOLINT

    const int N = n_tokens;

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;

    const auto & kv_self = lctx.kv_self;

    GGML_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_layer     = hparams.n_layer;
    const int64_t n_ctx       = hparams.n_ctx;
    const int64_t n_head      = hparams.n_head;
    const int64_t n_head_kv   = hparams.n_head_kv;
    const int64_t n_embd_head = hparams.n_embd_head();
    const int64_t n_embd_gqa  = hparams.n_embd_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_rot);

    const float freq_base    = hparams.rope_freq_base;
    const float freq_scale   = hparams.rope_freq_scale;
    const float norm_rms_eps = hparams.f_norm_rms_eps;

    const int n_gpu_layers = model.n_gpu_layers;

    auto & buf_compute = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.data,
        /*.no_alloc   =*/ false,
    };

    params.no_alloc = true;

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    if (tokens) {
        struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);

        ggml_allocr_alloc(lctx.alloc, inp_tokens);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inp_tokens->data, tokens, N*ggml_element_size(inp_tokens));
        }
        ggml_set_name(inp_tokens, "inp_tokens");

        inpL = ggml_get_rows(ctx0, model.tok_embeddings, inp_tokens);
    } else {
#ifdef GGML_USE_MPI
        GGML_ASSERT(false && "not implemented");
#endif

        inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N);

        ggml_allocr_alloc(lctx.alloc, inpL);
        if (!ggml_allocr_is_measure(lctx.alloc)) {
            memcpy(inpL->data, embd, N * n_embd * ggml_element_size(inpL));
        }
    }

    const int i_gpu_start = n_layer - n_gpu_layers;
    (void) i_gpu_start;

    // offload functions set the tensor output backend to GPU
    // tensors are GPU-accelerated if any input or the output has been offloaded
    //
    // with the low VRAM option VRAM scratch is disabled in llama_load_model_internal
    // in that case ggml_cuda_assign_buffers has no effect
    offload_func_t offload_func_nr = llama_nop; // nr = non-repeating
    offload_func_t offload_func_kq = llama_nop;
    offload_func_t offload_func_v  = llama_nop;

#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > n_layer) {
        offload_func_nr = ggml_cuda_assign_buffers_no_alloc;
    }
    if (n_gpu_layers > n_layer + 1) {
        offload_func_v  = ggml_cuda_assign_buffers_no_alloc;
    }
    if (n_gpu_layers > n_layer + 2) {
        offload_func_kq = ggml_cuda_assign_buffers_no_alloc;
    }
#endif // GGML_USE_CUBLAS

    struct ggml_tensor * KQ_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_allocr_alloc(lctx.alloc, KQ_scale);
    if (!ggml_allocr_is_measure(lctx.alloc)) {
        ggml_set_f32(KQ_scale, 1.0f/sqrtf(float(n_embd)/n_head));
    }
    ggml_set_name(KQ_scale, "1/sqrt(n_embd_head)");

    for (int il = 0; il < n_layer; ++il) {
        ggml_format_name(inpL, "layer_inp_%d", il);

        offload_func_t offload_func = llama_nop;

#ifdef GGML_USE_CUBLAS
        if (il >= i_gpu_start) {
            offload_func = ggml_cuda_assign_buffers_no_alloc;
        }
#endif // GGML_USE_CUBLAS

        struct ggml_tensor * inpSA = inpL;

        // norm
        {
            cur = ggml_rms_norm(ctx0, inpL, norm_rms_eps);
            offload_func(cur);
            ggml_set_name(cur, "rms_norm_0");

            // cur = cur*attn_norm(broadcasted)
            cur = ggml_mul(ctx0, cur, model.layers[il].attn_norm);
            offload_func(cur);
            ggml_set_name(cur, "attention_norm_0");
        }

        // self-attention
        {
            // compute Q and K and RoPE them
            struct ggml_tensor * tmpk = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            offload_func_kq(tmpk);
            ggml_set_name(tmpk, "tmpk");

            struct ggml_tensor * tmpq = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            offload_func_kq(tmpq);
            ggml_set_name(tmpq, "tmpq");

            struct ggml_tensor * Kcur = ggml_rope_custom_inplace(ctx0, ggml_reshape_3d(ctx0, tmpk, n_embd_head, n_head_kv, N), n_past, n_embd_head, 0, 0, freq_base, freq_scale);
            offload_func_kq(Kcur);
            ggml_set_name(Kcur, "Kcur");

            struct ggml_tensor * Qcur = ggml_rope_custom_inplace(ctx0, ggml_reshape_3d(ctx0, tmpq, n_embd_head, n_head, N),    n_past, n_embd_head, 0, 0, freq_base, freq_scale);
            offload_func_kq(Qcur);
            ggml_set_name(Qcur, "Qcur");

            // store key and value to memory
            {
                // compute the transposed [N, n_embd] V matrix

                struct ggml_tensor * tmpv = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                offload_func_v(tmpv);
                ggml_set_name(tmpv, "tmpv");

                struct ggml_tensor * Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, tmpv, n_embd_gqa, N));
                offload_func_v(Vcur);
                ggml_set_name(Vcur, "Vcur");

                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd_gqa, (ggml_element_size(kv_self.k)*n_embd_gqa)*(il*n_ctx + n_past));
                offload_func_kq(k);
                ggml_set_name(k, "k");

                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v, N, n_embd_gqa,
                        (   n_ctx)*ggml_element_size(kv_self.v),
                        (il*n_ctx)*ggml_element_size(kv_self.v)*n_embd_gqa + n_past*ggml_element_size(kv_self.v));
                offload_func_v(v);
                ggml_set_name(v, "v");

                // important: storing RoPE-ed version of K in the KV cache!
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            }

            struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            offload_func_kq(Q);
            ggml_set_name(Q, "Q");

            struct ggml_tensor * K =
                ggml_view_3d(ctx0, kv_self.k,
                        n_embd_head, n_past + N, n_head_kv,
                        ggml_element_size(kv_self.k)*n_embd_gqa,
                        ggml_element_size(kv_self.k)*n_embd_head,
                        ggml_element_size(kv_self.k)*n_embd_gqa*n_ctx*il);
            offload_func_kq(K);
            ggml_set_name(K, "K");

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            offload_func_kq(KQ);
            ggml_set_name(KQ, "KQ");

            // KQ_scaled = KQ / sqrt(n_embd_head)
            // KQ_scaled shape [n_past + N, N, n_head, 1]
            struct ggml_tensor * KQ_scaled = ggml_scale_inplace(ctx0, KQ, KQ_scale);
            offload_func_kq(KQ_scaled);
            ggml_set_name(KQ_scaled, "KQ_scaled");

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);
            offload_func_kq(KQ_masked);
            ggml_set_name(KQ_masked, "KQ_masked");

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);
            offload_func_v(KQ_soft_max);
            ggml_set_name(KQ_soft_max, "KQ_soft_max");

            // split cached V into n_head heads
            struct ggml_tensor * V =
                ggml_view_3d(ctx0, kv_self.v,
                        n_past + N, n_embd_head, n_head_kv,
                        ggml_element_size(kv_self.v)*n_ctx,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_head,
                        ggml_element_size(kv_self.v)*n_ctx*n_embd_gqa*il);
            offload_func_v(V);
            ggml_set_name(V, "V");

#if 1
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            offload_func_v(KQV);
            ggml_set_name(KQV, "KQV");
#else
            // make V contiguous in memory to speed up the matmul, however we waste time on the copy
            // on M1 this is faster for the perplexity computation, but ~5% slower for the single-token generation
            // is there a better way?
            struct ggml_tensor * V_cont = ggml_cpy(ctx0, V, ggml_new_tensor_3d(ctx0, kv_self.v->type, n_past + N, n_embd_head, n_head));
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_cont, KQ_soft_max);
#endif

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            offload_func_v(KQV_merged);
            ggml_set_name(KQV_merged, "KQV_merged");

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));
            offload_func_v(cur);
            ggml_set_name(cur, "KQV_merged_contiguous");

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_wo");
        }

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);
        offload_func(inpFF);
        ggml_set_name(inpFF, "inpFF");

        // feed-forward network
        {
            // norm
            {
                cur = ggml_rms_norm(ctx0, inpFF, norm_rms_eps);
                offload_func(cur);
                ggml_set_name(cur, "rms_norm_1");

                // cur = cur*ffn_norm(broadcasted)
                cur = ggml_mul(ctx0, cur, model.layers[il].ffn_norm);
                offload_func(cur);
                ggml_set_name(cur, "ffn_norm");
            }

            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model.layers[il].w3,
                    cur);
            offload_func(tmp);
            ggml_set_name(tmp, "result_w3");

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_w1");

            // SILU activation
            cur = ggml_silu(ctx0, cur);
            offload_func(cur);
            ggml_set_name(cur, "silu");

            cur = ggml_mul(ctx0, cur, tmp);
            offload_func(cur);
            ggml_set_name(cur, "silu_x_result_w3");

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
            offload_func(cur);
            ggml_set_name(cur, "result_w2");
        }

        cur = ggml_add(ctx0, cur, inpFF);
        offload_func(cur);
        ggml_set_name(cur, "inpFF_+_result_w2");

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    // norm
    {
        cur = ggml_rms_norm(ctx0, cur, norm_rms_eps);
        offload_func_nr(cur);
        ggml_set_name(cur, "rms_norm_2");

        // cur = cur*norm(broadcasted)
        cur = ggml_mul(ctx0, cur, model.output_norm);
        // offload_func_nr(cur); // TODO CPU + GPU mirrored backend
        ggml_set_name(cur, "result_norm");
    }

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);
    ggml_set_name(cur, "result_output");

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);

    return gf;
}

```