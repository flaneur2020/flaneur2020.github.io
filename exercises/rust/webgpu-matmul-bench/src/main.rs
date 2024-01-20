use rand::prelude::*;
use std::{borrow::Cow, cell::RefCell, time::Duration};
use wgpu::util::DeviceExt;

struct Workload {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    query_set: wgpu::QuerySet,
    query_set_idx: RefCell<usize>,
    query_set_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    dispatch_workgroups: (usize, usize, usize),
}

impl Workload {
    fn new(
        shader: &'static str,
        staging_buf_size: usize,
        dispatch_workgroups: (usize, usize, usize),
    ) -> Self {
        let (device, queue) = pollster::block_on(init_gpu());
        let module = unsafe {
            // create_shader_module_unchecked seems much faster than create_shader_module
            device.create_shader_module_unchecked(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
            })
        };

        // prepare the pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: "main",
        });

        // prepare the query set to track timestamps
        let query_set_len: usize = 1000;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count: query_set_len as u32,
        });
        let query_set_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (query_set_len * std::mem::size_of::<u64>()) as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // the cpu buffer to receive the result
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging buffer"),
            size: staging_buf_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            pipeline,
            query_set,
            query_set_buffer,
            staging_buffer,
            dispatch_workgroups,
            query_set_idx: RefCell::new(0),
        }
    }

    pub fn make_rand_buf(&self, elems: usize) -> (wgpu::Buffer, Vec<f32>) {
        let mut rng = rand::thread_rng();
        let data = (0..elems).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        (buf, data)
    }

    pub fn dump_durations(&self) -> Vec<Duration> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.resolve_query_set(
            &self.query_set,
            0..*self.query_set_idx.borrow() as u32,
            &self.query_set_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.query_set_buffer,
            0,
            &self.staging_buffer,
            0,
            self.query_set_buffer.size(),
        );
        self.queue.submit(Some(encoder.finish()));

        self.staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        self.device.poll(wgpu::Maintain::Wait);

        let timestamp_view = self
            .staging_buffer
            .slice(
                ..(std::mem::size_of::<u64>() * *self.query_set_idx.borrow() as usize)
                    as wgpu::BufferAddress,
            )
            .get_mapped_range();

        let timestamp_period = self.queue.get_timestamp_period() as u64;
        let timestamps: &[u64] = bytemuck::cast_slice(&timestamp_view);
        timestamps
            .chunks(2)
            .map(|w| (w[1] - w[0]) * timestamp_period)
            .map(Duration::from_nanos)
            .collect()
    }

    pub fn output(&self, output_buffer: wgpu::Buffer, output: &mut [f32]) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &self.staging_buffer,
            0,
            (output.len() * std::mem::size_of::<f32>()) as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        // await the result from staging buffer
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let staging_slice = self.staging_buffer.slice(..);
        staging_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            // Gets contents of buffer
            let data = staging_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            output.copy_from_slice(bytemuck::cast_slice(&data));

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            self.staging_buffer.unmap();
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}

async fn init_gpu() -> (wgpu::Device, wgpu::Queue) {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();
    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    println!("adapter: {:?}", adapter.get_info());
    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::TIMESTAMP_QUERY,
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();
    (device, queue)
}

// a: [m, n]
// b: [n, k]
// c: [m, k]
fn sgemm(
    workload: &Workload,
    m: usize,
    n: usize,
    k: usize,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    c: &wgpu::Buffer,
) {
    let buf_m = workload
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer c"),
            contents: bytemuck::cast_slice(&[m as u32, k as u32, n as u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE,
        });

    let bind_group_layout = workload.pipeline.get_bind_group_layout(0);
    let bind_group = workload
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_m.as_entire_binding(),
                },
            ],
        });

    // encode the commands into queue
    let mut encoder = workload
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut query_set_idx = workload.query_set_idx.borrow_mut();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &workload.query_set,
                beginning_of_pass_write_index: Some(*query_set_idx as u32),
                end_of_pass_write_index: Some((*query_set_idx + 1) as u32),
            }),
        });
        *query_set_idx += 2;

        pass.set_pipeline(&workload.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            workload.dispatch_workgroups.0 as u32,
            workload.dispatch_workgroups.1 as u32,
            workload.dispatch_workgroups.2 as u32,
        );
    }
    workload.queue.submit(Some(encoder.finish()));
}

fn load_gemm_workloads(m: usize, k: usize, n: usize) -> Vec<(&'static str, Workload)> {
    let staging_buf_size = (m * n) * std::mem::size_of::<f32>();

    let mut workloads = vec![];
    workloads.push((
        "gemm1",
        Workload::new(
            include_str!("../shaders/gemm/gemm1_naive.wgsl"),
            staging_buf_size,
            (m / 64, 1, 1),
        ),
    ));
    workloads.push((
        "gemm2",
        Workload::new(
            include_str!("../shaders/gemm/gemm2_naive.wgsl"),
            staging_buf_size,
            (m, n, 1),
        ),
    ));
    workloads.push((
        "gemm3",
        Workload::new(
            include_str!("../shaders/gemm/gemm3_naive.wgsl"),
            staging_buf_size,
            (m / 16, n / 16, 1),
        ),
    ));
    workloads.push((
        "gemm4",
        Workload::new(
            include_str!("../shaders/gemm/gemm4_basic_vectorized.wgsl"),
            staging_buf_size,
            (m / 8, n / 32, 1),
        ),
    ));
    workloads.push((
        "gemm5",
        Workload::new(
            include_str!("../shaders/gemm/gemm5_tiled.wgsl"),
            staging_buf_size,
            (m / 16, n / 16, 1),
        ),
    ));
    workloads.push((
        "gemm6",
        Workload::new(
            include_str!("../shaders/gemm/gemm6_tiled_vectorized.wgsl"),
            staging_buf_size,
            (m / 16, n / 16, 1),
        ),
    ));

    workloads
}

fn main() {
    let (m, k, n) = (1024, 1024, 1024);
    let workloads = load_gemm_workloads(m, k, n);
    let workload = &workloads
        .into_iter()
        .filter(|(name, w)| *name == "gemm6")
        .last()
        .unwrap()
        .1;

    let buf_a = workload.make_rand_buf(m * k).0;
    let buf_b = workload.make_rand_buf(k * n).0;
    let buf_c = workload.make_rand_buf(m * n).0;

    // prewarm
    sgemm(&workload, m, n, k, &buf_a, &buf_b, &buf_c);
    sgemm(&workload, m, n, k, &buf_a, &buf_b, &buf_c);
    workload.device.poll(wgpu::Maintain::Wait);

    let start_at = std::time::Instant::now();
    sgemm(&workload, m, n, k, &buf_a, &buf_b, &buf_c);
    sgemm(&workload, m, n, k, &buf_b, &buf_c, &buf_a);
    sgemm(&workload, m, n, k, &buf_c, &buf_a, &buf_b);
    sgemm(&workload, m, n, k, &buf_a, &buf_b, &buf_c);
    sgemm(&workload, m, n, k, &buf_a, &buf_b, &buf_c);
    sgemm(&workload, m, n, k, &buf_a, &buf_b, &buf_c);
    sgemm(&workload, m, n, k, &buf_a, &buf_b, &buf_c);
    workload.device.poll(wgpu::Maintain::Wait);

    let walltime_secs = start_at.elapsed().as_secs_f64();
    println!("walltime_elapsed: {}", walltime_secs);

    let durations = workload.dump_durations();
    for (i, duration) in durations.iter().enumerate() {
        let gflops =
            (2 * (m * k * n) / 1024 / 1024 / 1024) as f64 / (duration.as_nanos() as f64 / 1e9);
        println!(
            "{} elapsed (ns): {} gflops: {:.2}",
            i,
            duration.as_nanos(),
            gflops
        );
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_relative_eq, relative_eq};

    use crate::{load_gemm_workloads, sgemm, Workload};

    fn vanilla_matmul(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0;
                for ki in 0..k {
                    sum += a[mi * k + ki] * b[ki * n + ni];
                }
                c[mi * n + ni] = sum;
            }
        }
    }

    #[test]
    fn test_gemm_correctness() {
        let (m, k, n) = (256, 256, 256);
        let workloads = load_gemm_workloads(m, k, n);
        for (name, workload) in workloads {
            let (buf_a, vec_a) = workload.make_rand_buf(m * k);
            let (buf_b, vec_b) = workload.make_rand_buf(k * n);
            let (buf_c, mut vec_c) = workload.make_rand_buf(m * n);

            sgemm(&workload, m, n, k, &buf_a, &buf_b, &buf_c);
            workload.output(buf_c, &mut vec_c);

            let mut vec_c2 = vec![0.0; m * n];
            vanilla_matmul(m, k, n, &vec_a, &vec_b, &mut vec_c2);

            println!("workload: {}", name);
            assert_relative_eq!(vec_c[0..800], vec_c2[0..800], epsilon = 1e-1);
            assert!(
                relative_eq!(vec_c[..], vec_c2[..], epsilon = 1e-1),
                "workload {}",
                name
            );
        }
    }
}
