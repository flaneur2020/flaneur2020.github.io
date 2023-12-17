use std::{borrow::Cow, cell::RefCell};
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
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
        });

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

    pub fn record_timestamp(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut query_idx = self.query_set_idx.borrow_mut();
        encoder.write_timestamp(&self.query_set, *query_idx as u32);
        *query_idx += 1;
    }

    pub fn dump_timestamps(&self) -> Vec<u64> {
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

        let timestamps: &[u64] = bytemuck::cast_slice(&timestamp_view);
        timestamps.to_vec()
    }

    pub async fn output(&self, output_buffer: wgpu::Buffer, output: &mut [f32]) {
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
            contents: bytemuck::cast_slice(&[m as u32, n as u32, k as u32]),
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
        *query_set_idx += 1;

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

fn main() {
    let (m, n, k) = (1024, 1024, 1024);
    let a = vec![1.0; m * n];
    let b = vec![2.0; n * k];
    let mut c = vec![0.0; m * k];

    let staging_buf_size = (m * k) * std::mem::size_of::<f32>();

    let workload_kind = "matmul_naive";
    let workload = match workload_kind {
        "matmul_naive" => Workload::new(
            include_str!("../shaders/matmul_naive.wgsl"),
            staging_buf_size,
            (m / 64, 1, 1),
        ),
        "matmul_split_work" => Workload::new(
            include_str!("../shaders/matmul_split_work.wgsl"),
            staging_buf_size,
            (m, n, 1),
        ),
        _ => panic!("unknown workload kind: {}", workload_kind),
    };

    let buf_a = workload
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer a"),
            contents: bytemuck::cast_slice(&a),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_b = workload
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer b"),
            contents: bytemuck::cast_slice(&b),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_c = workload
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer c"),
            contents: bytemuck::cast_slice(&c),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    // prewarm
    sgemm(&workload, m, n, k, &buf_a, &buf_b, &buf_c);
    workload.device.poll(wgpu::Maintain::Wait);

    let start_at = std::time::Instant::now();
    sgemm(&workload, m, n, k, &buf_a, &buf_b, &buf_c);
    sgemm(&workload, m, n, k, &buf_b, &buf_c, &buf_a);
    sgemm(&workload, m, n, k, &buf_c, &buf_a, &buf_b);
    sgemm(&workload, m, n, k, &buf_a, &buf_b, &buf_c);
    workload.device.poll(wgpu::Maintain::Wait);

    let walltime_secs = start_at.elapsed().as_secs_f64();
    let timestamps = workload.dump_timestamps();
    for i in 0..timestamps.len() / 2 {
        let timestamp_period = workload.queue.get_timestamp_period() as f64;
        println!(
            "duration (ns) {}: {}",
            i,
            (timestamps[i * 2 + 1] - timestamps[i * 2]) as f64 * timestamp_period
        );
    }

    let gputime_secs = (timestamps.last().unwrap() - timestamps[2]) as f64 / 1e9;
    let avg_gflops = (4 * 2 * (m * k * n) / 1024 / 1024 / 1024) as f64 / gputime_secs;

    println!(
        "elapsed: {}, gpu elapsed: {} avg flops: {:.2}G",
        walltime_secs, gputime_secs, avg_gflops
    );
}
