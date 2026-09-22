//! Render SAM3 masks for every JPEG/PNG in one folder. Inference stays in Rust.
#[path = "../sam3/clip_tokenizer.rs"]
mod clip_tokenizer;

use anyhow::{Context, Result, ensure};
use clip_tokenizer::ClipTokenizer;
use image::{ImageReader, Rgb, RgbImage, imageops::FilterType};
use safe_inference::{
    Backend, BackendSelection, CudaOptions, OnnxOptions, OnnxRuntime, Tensor, TensorBuffer,
    TensorData, TensorView,
};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

// Change the prompt here. Rust tokenizes it from the bundled CLIP BPE vocabulary.
const PROMPT: &str = "plants";
const SIZE: usize = 896;
const GRID: usize = 256;
const MIN_CONFIDENCE: f32 = 0.25;

fn tensor_f32<'a>(tensors: &'a [Tensor], name: &str) -> Result<(&'a [usize], &'a [f32])> {
    let tensor = tensors
        .iter()
        .find(|item| item.name == name)
        .with_context(|| format!("missing {name}"))?;
    let TensorBuffer::F32(values) = &tensor.data else {
        anyhow::bail!("{name} is not FP32")
    };
    Ok((&tensor.shape, values))
}

fn preprocess(path: &Path) -> Result<(RgbImage, Vec<f32>)> {
    let image = ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?
        .to_rgb8();
    let resized = image::imageops::resize(&image, SIZE as u32, SIZE as u32, FilterType::Triangle);
    let mut chw = vec![0.0_f32; 3 * SIZE * SIZE];
    for (pixel_index, pixel) in resized.pixels().enumerate() {
        for channel in 0..3 {
            chw[channel * SIZE * SIZE + pixel_index] = (f32::from(pixel[channel]) - 127.5) / 127.5;
        }
    }
    Ok((image, chw))
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn mask_value(mask: &[f32], x: u32, y: u32, width: u32, height: u32) -> f32 {
    let sx = ((x as f32 + 0.5) * GRID as f32 / width as f32 - 0.5).clamp(0.0, (GRID - 1) as f32);
    let sy = ((y as f32 + 0.5) * GRID as f32 / height as f32 - 0.5).clamp(0.0, (GRID - 1) as f32);
    let x0 = sx.floor() as usize;
    let y0 = sy.floor() as usize;
    let x1 = (x0 + 1).min(GRID - 1);
    let y1 = (y0 + 1).min(GRID - 1);
    let dx = sx - x0 as f32;
    let dy = sy - y0 as f32;
    let top = mask[y0 * GRID + x0] * (1.0 - dx) + mask[y0 * GRID + x1] * dx;
    let bottom = mask[y1 * GRID + x0] * (1.0 - dx) + mask[y1 * GRID + x1] * dx;
    top * (1.0 - dy) + bottom * dy
}

fn render(image: &RgbImage, mask: &[f32], output: &Path) -> Result<()> {
    let (width, height) = image.dimensions();
    let mut overlay = image.clone();
    for (x, y, pixel) in overlay.enumerate_pixels_mut() {
        if mask_value(mask, x, y, width, height) > 0.0 {
            let original = image.get_pixel(x, y);
            *pixel = Rgb([
                original[0] / 2,
                (u16::from(original[1]) / 2 + 127) as u8,
                original[2] / 2,
            ]);
        }
    }
    overlay
        .save(output)
        .with_context(|| format!("save {}", output.display()))?;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let image_dir = PathBuf::from(
        args.next()
            .context("usage: sam3_render <image-folder> <output-folder> [cuda|cpu]")?,
    );
    let output_dir = PathBuf::from(args.next().context("missing output folder")?);
    let backend = match args.next().as_deref() {
        None | Some("cuda") => Backend::Cuda,
        Some("cpu") => Backend::Cpu,
        _ => anyhow::bail!("backend must be cuda or cpu"),
    };
    ensure!(args.next().is_none(), "too many arguments");
    fs::create_dir_all(&output_dir)?;
    ensure!(
        fs::canonicalize(&image_dir)? != fs::canonicalize(&output_dir)?,
        "input and output folders must differ"
    );
    let model_dir = std::env::var_os("SAM3_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("sam3/mixed_cuda"));
    let tokens =
        ClipTokenizer::load(&model_dir.join("bpe_simple_vocab_16e6.txt.gz"))?.encode(PROMPT)?;
    let options = OnnxOptions {
        backend: BackendSelection::Require(backend),
        cuda: Some(CudaOptions {
            device_id: 0,
            tf32: true,
            cuda_graph: true,
        }),
        ..OnnxOptions::default()
    };
    let mut text = OnnxRuntime::load(model_dir.join("text_encoder.onnx"), options.clone())?;
    let text_start = Instant::now();
    let language = text.inference(&[TensorView {
        name: "tokens",
        shape: &[1, 32],
        data: TensorData::I64(&tokens),
    }])?;
    let text_time = text_start.elapsed();
    drop(text);
    let mut image_encoder =
        OnnxRuntime::load(model_dir.join("image_encoder.onnx"), options.clone())?;
    let mut decoder = OnnxRuntime::load(model_dir.join("text_decoder.onnx"), options)?;

    let mut images = fs::read_dir(&image_dir)?
        .map(|entry| entry.map(|item| item.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    images.retain(|path| {
        path.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| {
                    ["jpg", "jpeg", "png"].contains(&ext.to_ascii_lowercase().as_str())
                })
    });
    images.sort();
    ensure!(
        !images.is_empty(),
        "no JPEG or PNG files in {}",
        image_dir.display()
    );
    println!(
        "Text encoder (once): {:.1} ms",
        text_time.as_secs_f64() * 1_000.0
    );
    let image_count = images.len();
    let mut image_time = Duration::ZERO;
    let mut decoder_time = Duration::ZERO;
    for path in images {
        let (image, chw) = preprocess(&path)?;
        let image_start = Instant::now();
        let features =
            image_encoder.inference(&[TensorView::f32("image", &[1, 3, SIZE, SIZE], &chw)])?;
        let image_elapsed = image_start.elapsed();
        let inputs = decoder
            .info()
            .inputs
            .iter()
            .map(|spec| {
                features
                    .iter()
                    .chain(&language)
                    .find(|tensor| tensor.name == spec.name)
                    .map(Tensor::view)
                    .with_context(|| format!("missing decoder input {}", spec.name))
            })
            .collect::<Result<Vec<_>>>()?;
        let decoder_start = Instant::now();
        let decoded = decoder.inference(&inputs)?;
        let decoder_elapsed = decoder_start.elapsed();
        image_time += image_elapsed;
        decoder_time += decoder_elapsed;
        let (logit_shape, logits) = tensor_f32(&decoded, "pred_logits")?;
        let (mask_shape, masks) = tensor_f32(&decoded, "pred_masks")?;
        let (_, presence) = tensor_f32(&decoded, "presence_logit")?;
        ensure!(
            logit_shape.len() == 3 && logit_shape[0] == 1 && logit_shape[2] == 1,
            "unexpected logit shape"
        );
        let count = logit_shape[1];
        ensure!(
            mask_shape == [1, count, GRID, GRID],
            "unexpected mask shape"
        );
        let mut union = vec![f32::NEG_INFINITY; GRID * GRID];
        let mut selected = 0;
        for index in 0..count {
            if sigmoid(logits[index]) * sigmoid(presence[0]) < MIN_CONFIDENCE {
                continue;
            }
            selected += 1;
            let mask = &masks[index * GRID * GRID..(index + 1) * GRID * GRID];
            for (destination, &source) in union.iter_mut().zip(mask) {
                *destination = destination.max(source);
            }
        }
        let name = path
            .file_name()
            .context("image path has no filename")?
            .to_string_lossy();
        let output = output_dir.join(format!("{name}.overlay.png"));
        render(&image, &union, &output)?;
        let forward = image_elapsed + decoder_elapsed;
        println!(
            "{}: {selected} masks; inference {:.1} ms (image {:.1} ms, decoder {:.1} ms); {}",
            path.display(),
            forward.as_secs_f64() * 1_000.0,
            image_elapsed.as_secs_f64() * 1_000.0,
            decoder_elapsed.as_secs_f64() * 1_000.0,
            output.display()
        );
    }
    let total = image_time + decoder_time;
    println!(
        "Average inference: {:.1} ms/image ({:.2} images/s; image {:.1} ms, decoder {:.1} ms)",
        total.as_secs_f64() * 1_000.0 / image_count as f64,
        image_count as f64 / total.as_secs_f64(),
        image_time.as_secs_f64() * 1_000.0 / image_count as f64,
        decoder_time.as_secs_f64() * 1_000.0 / image_count as f64,
    );
    Ok(())
}
