use std::{borrow::Cow, future::Future};

use wgpu::util::DeviceExt;

struct GpuCompute {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    staging_buffer: wgpu::Buffer,
}

impl GpuCompute {
    async fn new(shader: &'static str, staging_buf_size: usize) -> Self {
        let (device, queue) = init_gpu().await;
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
    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();
    (device, queue)
}

async fn compute_exp(input: &[f32], output: &mut [f32]) {
    let buf_size = input.len() * std::mem::size_of::<f32>();
    let gpu = GpuCompute::new(include_str!("../shaders/exp.wgsl"), buf_size).await;

    let input_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input buffer"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    let bind_group_layout = gpu.pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: input_buffer.as_entire_binding(),
        }],
    });

    // encode the commands into queue
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.insert_debug_marker("compute exp iterations");
        pass.dispatch_workgroups(input.len() as u32 / 64 + 1, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }
    gpu.queue.submit(Some(encoder.finish()));

    // await the result from staging buffer
    gpu.output(input_buffer, output).await;
}

fn main() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; input.len()];
    pollster::block_on(compute_exp(&input, &mut output));
    println!("{:?}", output);
}
