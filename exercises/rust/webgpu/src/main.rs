use std::borrow::Cow;

use wgpu::util::DeviceExt;

fn compute_exp_once(
    device: &mut wgpu::Device,
    queue: &wgpu::Queue,
    input: &[f32],
    output: &mut [f32],
) {
    // load wsgl module
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/exp.wgsl"))),
    });

    // prepare the buffers
    let buf_size = std::mem::size_of_val(input) as wgpu::BufferAddress;
    let (input_buffer, output_buffer, staging_buffer) = {
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input Buffer"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (input_buffer, output_buffer, staging_buffer)
    };

    // prepare the pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: "main",
    });

    // pass the bind groups
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // encode the commands into queue
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.insert_debug_marker("compute exp iterations");
        pass.dispatch_workgroups(input.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, buf_size);
    queue.submit(Some(encoder.finish()));

    // await the result
    let (tx, rx) = std::sync::mpsc::channel();
    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);

    if let Ok(Ok(())) = rx.recv() {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        output.copy_from_slice(bytemuck::cast_slice(&data));

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        // output_buffer.unmap(); // Unmaps buffer from memory
    } else {
        panic!("failed to run compute on gpu!")
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
    let (mut device, mut queue) = init_gpu().await;
    compute_exp_once(&mut device, &mut queue, input, output);
}

fn main() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; input.len()];
    pollster::block_on(compute_exp(&input, &mut output));
    println!("input: {:?}", input);
    println!("ouput: {:?}", output);
}
