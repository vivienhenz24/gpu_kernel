use metal::*;
use std::path::Path;
use std::time::Instant;

const NUM_SAMPLES: u64 = 4;
const METAL_SHADER: &str = include_str!("ai.metal");

pub fn unbound_vivien() {
    let input_path = Path::new("public/asset.jpg");
    let output_path = Path::new("public/asset_blur.png");
    let input_image = image::open(input_path)
        .expect("Failed to open public/asset.jpg")
        .to_rgba8();
    let (width, height) = input_image.dimensions();
    let input_bytes = input_image.into_raw();

    let device = Device::system_default()
        .expect("No Metal device found. This code requires macOS with Metal support.");

    let library = device
        .new_library_with_source(METAL_SHADER, &CompileOptions::new())
        .expect("Failed to compile Metal shader");
    let blur_h = library
        .get_function("gaussian_blur_h", None)
        .expect("Failed to get kernel function 'gaussian_blur_h'");
    let blur_v = library
        .get_function("gaussian_blur_v", None)
        .expect("Failed to get kernel function 'gaussian_blur_v'");
    let pipeline_h = device
        .new_compute_pipeline_state_with_function(&blur_h)
        .expect("Failed to create compute pipeline (horizontal)");
    let pipeline_v = device
        .new_compute_pipeline_state_with_function(&blur_v)
        .expect("Failed to create compute pipeline (vertical)");

    let command_queue = device.new_command_queue();

    let texture_desc = TextureDescriptor::new();
    texture_desc.set_texture_type(MTLTextureType::D2);
    texture_desc.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    texture_desc.set_width(width as u64);
    texture_desc.set_height(height as u64);
    texture_desc.set_storage_mode(MTLStorageMode::Shared);
    texture_desc.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);

    let input_tex = device.new_texture(&texture_desc);
    let temp_tex = device.new_texture(&texture_desc);
    let output_tex = device.new_texture(&texture_desc);

    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };
    input_tex.replace_region(
        region,
        0,
        input_bytes.as_ptr() as *const _,
        (width * 4) as u64,
    );

    let command_buffer = command_queue.new_command_buffer();

    let counter_sampling_point = MTLCounterSamplingPoint::AtStageBoundary;
    let counter_sample_buffer = if device.supports_counter_sampling(counter_sampling_point) {
        create_counter_sample_buffer(&device)
    } else {
        None
    };
    let resolved_sample_buffer = counter_sample_buffer.as_ref().map(|_| {
        device.new_buffer(
            (std::mem::size_of::<u64>() * NUM_SAMPLES as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    });

    {
        let encoder = if let Some(sample_buffer) = counter_sample_buffer.as_ref() {
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let attachment = compute_pass_descriptor
                .sample_buffer_attachments()
                .object_at(0)
                .expect("Missing sample buffer attachment");
            attachment.set_sample_buffer(sample_buffer.as_ref());
            attachment.set_start_of_encoder_sample_index(0);
            attachment.set_end_of_encoder_sample_index(1);
            command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor)
        } else {
            command_buffer.new_compute_command_encoder()
        };
        encoder.set_compute_pipeline_state(&pipeline_h);
        encoder.set_texture(0, Some(&input_tex));
        encoder.set_texture(1, Some(&temp_tex));
        dispatch_2d(encoder, &pipeline_h, width as u64, height as u64);
        encoder.end_encoding();
    }

    {
        let encoder = if let Some(sample_buffer) = counter_sample_buffer.as_ref() {
            let compute_pass_descriptor = ComputePassDescriptor::new();
            let attachment = compute_pass_descriptor
                .sample_buffer_attachments()
                .object_at(0)
                .expect("Missing sample buffer attachment");
            attachment.set_sample_buffer(sample_buffer.as_ref());
            attachment.set_start_of_encoder_sample_index(2);
            attachment.set_end_of_encoder_sample_index(3);
            command_buffer.compute_command_encoder_with_descriptor(compute_pass_descriptor)
        } else {
            command_buffer.new_compute_command_encoder()
        };
        encoder.set_compute_pipeline_state(&pipeline_v);
        encoder.set_texture(0, Some(&temp_tex));
        encoder.set_texture(1, Some(&output_tex));
        dispatch_2d(encoder, &pipeline_v, width as u64, height as u64);
        encoder.end_encoding();
    }

    if let (Some(sample_buffer), Some(resolve_buffer)) =
        (counter_sample_buffer.as_ref(), resolved_sample_buffer.as_ref())
    {
        resolve_samples_into_buffer(command_buffer, sample_buffer.as_ref(), resolve_buffer);
    }

    let mut cpu_start = 0;
    let mut gpu_start = 0;
    if counter_sample_buffer.is_some() {
        device.sample_timestamps(&mut cpu_start, &mut gpu_start);
    }
    let cpu_wait_start = Instant::now();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let cpu_ms = cpu_wait_start.elapsed().as_secs_f64() * 1000.0;

    let mut output_bytes = vec![0u8; (width * height * 4) as usize];
    output_tex.get_bytes(
        output_bytes.as_mut_ptr() as *mut _,
        (width * 4) as u64,
        region,
        0,
    );

    let output_image =
        image::RgbaImage::from_raw(width, height, output_bytes).expect("Invalid output image");
    output_image
        .save(output_path)
        .expect("Failed to write public/asset_blur.png");

    if let Some(ref resolve_buffer) = resolved_sample_buffer {
        let mut cpu_end = 0;
        let mut gpu_end = 0;
        device.sample_timestamps(&mut cpu_end, &mut gpu_end);
        let gpu_us = gpu_duration_us(resolve_buffer, cpu_start, cpu_end, gpu_start, gpu_end);
        println!(
            "Wrote {} (GPU time: {:.3} ms | CPU wait time: {:.3} ms)",
            output_path.display(),
            gpu_us / 1000.0,
            cpu_ms
        );
    } else {
        println!(
            "Wrote {} (CPU wait time: {:.3} ms, GPU counters not supported)",
            output_path.display(),
            cpu_ms
        );
    }
}

fn dispatch_2d(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineStateRef,
    width: u64,
    height: u64,
) {
    let threads_w = pipeline.thread_execution_width();
    let mut threads_h = pipeline.max_total_threads_per_threadgroup() / threads_w;
    if threads_h == 0 {
        threads_h = 1;
    }
    if threads_h > 16 {
        threads_h = 16;
    }
    let threads_per_group = MTLSize::new(threads_w, threads_h, 1);
    let groups = MTLSize::new(
        (width + threads_w - 1) / threads_w,
        (height + threads_h - 1) / threads_h,
        1,
    );
    encoder.dispatch_thread_groups(groups, threads_per_group);
}

fn create_counter_sample_buffer(device: &Device) -> Option<CounterSampleBuffer> {
    let counter_sets = device.counter_sets();
    let timestamp_counter = counter_sets.iter().find(|cs| cs.name() == "timestamp")?;
    let desc = CounterSampleBufferDescriptor::new();
    desc.set_storage_mode(MTLStorageMode::Shared);
    desc.set_sample_count(NUM_SAMPLES);
    desc.set_counter_set(timestamp_counter);
    device
        .new_counter_sample_buffer_with_descriptor(&desc)
        .ok()
}

fn resolve_samples_into_buffer(
    command_buffer: &CommandBufferRef,
    counter_sample_buffer: &CounterSampleBufferRef,
    destination_buffer: &BufferRef,
) {
    let blit_encoder = command_buffer.new_blit_command_encoder();
    blit_encoder.resolve_counters(
        counter_sample_buffer,
        metal::NSRange::new(0_u64, NUM_SAMPLES),
        destination_buffer,
        0_u64,
    );
    blit_encoder.end_encoding();
}

fn gpu_duration_us(
    resolved_sample_buffer: &BufferRef,
    cpu_start: u64,
    cpu_end: u64,
    gpu_start: u64,
    gpu_end: u64,
) -> f64 {
    let samples = unsafe {
        std::slice::from_raw_parts(
            resolved_sample_buffer.contents() as *const u64,
            NUM_SAMPLES as usize,
        )
    };
    let cpu_time_span = cpu_end - cpu_start;
    let gpu_time_span = gpu_end - gpu_start;
    let pass1 = microseconds_between_begin(samples[0], samples[1], gpu_time_span, cpu_time_span);
    let pass2 = microseconds_between_begin(samples[2], samples[3], gpu_time_span, cpu_time_span);
    pass1 + pass2
}

fn microseconds_between_begin(begin: u64, end: u64, gpu_time_span: u64, cpu_time_span: u64) -> f64 {
    if end <= begin || gpu_time_span == 0 || cpu_time_span == 0 {
        return 0.0;
    }
    let time_span = (end as f64) - (begin as f64);
    let nanoseconds = time_span / (gpu_time_span as f64) * (cpu_time_span as f64);
    nanoseconds / 1000.0
}
