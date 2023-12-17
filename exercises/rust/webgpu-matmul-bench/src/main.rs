use std::borrow::Cow;
use wgpu::util::DeviceExt;

struct Workload {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    staging_buffer: wgpu::Buffer,
}

impl Workload {
    fn new(shader: &'static str, staging_buf_size: usize) -> Self {
        let (device, queue) = pollster::block_on(init_gpu());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: "main",
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
            staging_buffer,
        }
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
                required_features: wgpu::Features::empty(),
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
async fn sgemm(
    gpu: &Workload,
    m: usize,
    n: usize,
    k: usize,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    c: &wgpu::Buffer,
) {
    let buf_m = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer c"),
            contents: bytemuck::cast_slice(&[m as u32, n as u32, k as u32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE,
        });

    let bind_group_layout = gpu.pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(m as u32, k as u32, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

fn main() {
    let (m, n, k) = (1024, 1024, 1024);
    let a = vec![1.0; m * n];
    let b = vec![2.0; n * k];
    let mut c = vec![0.0; m * k];

    let staging_buf_size = (m * k) * std::mem::size_of::<f32>();

    let gpu = Workload::new(
        include_str!("../shaders/matmul_split_work.wgsl"),
        staging_buf_size,
    );

    let buf_a = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer a"),
            contents: bytemuck::cast_slice(&a),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_b = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer b"),
            contents: bytemuck::cast_slice(&b),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_c = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("buffer c"),
            contents: bytemuck::cast_slice(&c),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    // prewarm
    pollster::block_on(sgemm(&gpu, m, n, k, &buf_a, &buf_b, &buf_c));
    gpu.device.poll(wgpu::Maintain::Wait);

    let start_at = std::time::Instant::now();
    pollster::block_on(async {
        sgemm(&gpu, m, n, k, &buf_a, &buf_b, &buf_c).await;
        sgemm(&gpu, m, n, k, &buf_b, &buf_c, &buf_a).await;
        sgemm(&gpu, m, n, k, &buf_c, &buf_a, &buf_b).await;
        sgemm(&gpu, m, n, k, &buf_a, &buf_b, &buf_c).await;
        gpu.output(buf_c, &mut c).await;
    });

    let duration_in_secs = start_at.elapsed().as_secs_f64();

    let gflops = (4 * 2 * (m * k * n) / 1024 / 1024 / 1024) as f64 / duration_in_secs;
    println!("elapsed: {} flops: {}G", duration_in_secs, gflops);

    // await the result from staging buffer
}
