use half::f16;

#[repr(C, packed)]
#[derive(Debug, Clone)]
pub struct BlockQ8_0 {
    d: f16,       // delta
    qs: [i8; 32], // quants
}

#[link(name = "ggml")]
extern "C" {
    fn ggml_vec_dot_q8_0_q8_0(
        n: i32,               // number of elements
        s: *mut f32,          // result
        bs: usize,            // not used?
        vx: *const BlockQ8_0, // binary of quantized vec x
        bx: usize,            // not used?
        vy: *const BlockQ8_0, // binary of quantized vec y
        by: usize,            // not used?
        nrc: i32,             // always 1?
    );
}

fn vec_dot_q8_ggml(n: i32, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let mut result: f32 = 0.0;
    unsafe {
        ggml_vec_dot_q8_0_q8_0(
            n,
            &mut result as *mut f32,
            0,
            x.as_ptr(),
            0,
            y.as_ptr(),
            0,
            1,
        );
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};

    // generate a random vector of BlockQ8_0
    fn gen_rand_block_q8_0() -> BlockQ8_0 {
        let mut rng = thread_rng();
        let d: f32 = rng.gen_range(0.0..2.0);
        let mut qs: [i8; 32] = [0; 32];
        for i in 0..32 {
            qs[i] = rng.gen::<i8>();
        }
        BlockQ8_0 {
            d: f16::from_f32(d),
            qs,
        }
    }

    fn gen_rand_block_q8_0_vec(n: usize) -> Vec<BlockQ8_0> {
        let mut v: Vec<BlockQ8_0> = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(gen_rand_block_q8_0());
        }
        v
    }

    #[test]
    fn test_vec_dot_q8_ggml() {
        let v1 = vec![
            BlockQ8_0 {
                d: f16::from_f32(1.0),
                qs: [1; 32],
            },
            BlockQ8_0 {
                d: f16::from_f32(1.0),
                qs: [1; 32],
            },
        ];
        let v2 = vec![
            BlockQ8_0 {
                d: f16::from_f32(1.0),
                qs: [2; 32],
            },
            BlockQ8_0 {
                d: f16::from_f32(1.0),
                qs: [2; 32],
            },
        ];
        let result = vec_dot_q8_ggml(64, &v1, &v2);
        assert_eq!(result, 128.0);
    }
}
