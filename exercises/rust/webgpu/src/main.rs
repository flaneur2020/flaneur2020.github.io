use std::{borrow::Cow, future::Future};

use wgpu::util::DeviceExt;

struct ComputeExp {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    buf_size: u64,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
}

impl ComputeExp {
    async fn new(buf_len: usize) -> Self {
        let (device, queue) = init_gpu().await;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/exp.wgsl"))),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: "main",
        });

        let buf_size = (buf_len * std::mem::size_of::<f32>()) as u64;

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output buffer"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // the cpu buffer to receive the result
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging buffer"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            device,
            queue,
            pipeline,
            buf_size,
            output_buffer,
            staging_buffer,
        }
    }

    pub async fn enqueue(&mut self, input: &[f32]) {
        assert!(input.len() == self.buf_size as usize / std::mem::size_of::<f32>());

        // setup input buffer
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input buffer"),
                contents: bytemuck::cast_slice(input),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // pass the bind groups
        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.output_buffer.as_entire_binding(),
                },
            ],
        });

        // encode the commands into queue
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.insert_debug_marker("compute exp iterations");
            pass.dispatch_workgroups(input.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
        }
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.buf_size,
        );
        self.queue.submit(Some(encoder.finish()));
    }

    pub async fn output(&self, output: &mut [f32]) {
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

fn main() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; input.len()];
    pollster::block_on(async {
        let mut compute = ComputeExp::new(input.len()).await;
        compute.enqueue(&input).await;
        compute.output(&mut output).await;
    });
    println!("{:?}", output);
}
