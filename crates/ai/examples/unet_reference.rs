//! # unet_reference
//!
//! **Author:** Joachim Danmayr
//! **Date:** 2026-11-25
//!
//! ## License
//! Copyright 2026 Joachim Danmayr.
//! Licensed under the **PolyForm Noncommercial License 1.0.0**.
//!
//! The burn migration's comparison against libtorch: a bioimage.io U-Net's
//! TorchScript (e.g. `tests/romantic-shark/weights-torchscript.pt`) -
//!
//! - its parameters into `<out>/weights.safetensors` (PyTorch's names, f32),
//! - the network (no padding, no pieces) on two inputs, normalized
//!   zero-mean-unit-variance (eps 1e-6, romantic-shark's preprocessing):
//!   the package's `test_input_0.npy` (256²) and `bench_unet`'s cells image
//!   (1024²), on the CPU and, when there is one, the GPU (CUDA, libtorch's
//!   default: TF32 in convolutions) -
//! - raw little-endian f32 files: `<out>/<input>.input.f32` (normalized),
//!   `<out>/<input>.libtorch-<cpu|gpu>.f32` (`[channels, H, W]`),
//! - and the engine as the app ran it (`evanalyzer_ai::infer`, CPU: mirror
//!   padding, pieces of 2048, foreground channel 0, boundary channel 1):
//!   `<out>/<input>.engine.f32` (foreground plane, then boundary plane).
//!
//!   cargo build-linux -p evanalyzer_ai --features test-support --example unet_reference
//!   LD_LIBRARY_PATH=$PWD/libs/libtorch/lib:$PWD/libs/nvidia-cu13/lib \
//!     LD_PRELOAD=$PWD/libs/libtorch/lib/libtorch_cuda.so \
//!     target/x86_64-unknown-linux-gnu/release/examples/unet_reference <torchscript> <package dir> <out dir>

use evanalyzer_ai::{DeviceSelection, EngineConfig, Image, Pieces, Request, Response, Task};
use std::path::Path;
use tch::{CModule, Device, Kind, Tensor};

/// `bench_unet`'s image: cells (bright squares) on a dim background.
fn cells(side: usize) -> Vec<f32> {
    (0..side * side)
        .map(|i| {
            let (x, y) = (i % side, i / side);
            let cell = (4..28).contains(&(x % 32)) && (4..28).contains(&(y % 32));
            if cell { 0.8 } else { 0.1 }
        })
        .collect()
}

/// Zero mean, unit variance over the whole image (bioimage.io's
/// `zero_mean_unit_variance`, eps 1e-6).
fn normalize(x: &Tensor) -> Tensor {
    let mean = x.mean(Kind::Float);
    let std = x.std(false);
    (x - mean) / (std + 1e-6)
}

fn write_f32(path: &Path, t: &Tensor) {
    let t = t
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .contiguous()
        .flatten(0, -1);
    let values: Vec<f32> = Vec::<f32>::try_from(&t).unwrap();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let script = Path::new(args.get(1).expect("TorchScript file"));
    let package = Path::new(args.get(2).expect("package dir (test_input_0.npy)"));
    let out = Path::new(args.get(3).expect("output dir"));
    std::fs::create_dir_all(out).unwrap();

    let cpu = CModule::load_on_device(script, Device::Cpu).unwrap();
    let params = cpu.named_parameters().unwrap();
    for (name, t) in &params {
        println!("{name} {:?}", t.size());
    }
    Tensor::write_safetensors(&params, out.join("weights.safetensors")).unwrap();

    let test = Tensor::read_npy(package.join("test_input_0.npy")).unwrap();
    let side = 1024;
    let bench = Tensor::from_slice(&cells(side)).reshape([1, 1, side as i64, side as i64]);
    let inputs = [
        ("test256", normalize(&test)),
        ("cells1024", normalize(&bench)),
    ];

    let mut devices = vec![("cpu", Device::Cpu)];
    if tch::Cuda::is_available() {
        devices.push(("gpu", Device::Cuda(0)));
    } else {
        println!("no CUDA: CPU only");
    }
    for (name, x) in &inputs {
        write_f32(&out.join(format!("{name}.input.f32")), x);
        for (dev_name, device) in &devices {
            let model = CModule::load_on_device(script, *device).unwrap();
            let y = tch::no_grad(|| model.forward_ts(&[x.to_device(*device)]).unwrap());
            println!("{name} libtorch-{dev_name}: {:?}", y.size());
            write_f32(&out.join(format!("{name}.libtorch-{dev_name}.f32")), &y);
        }
    }
    for (name, x) in &inputs {
        let size = x.size();
        let (h, w) = (size[2] as usize, size[3] as usize);
        let flat = x.contiguous().flatten(0, -1);
        let data: Vec<f32> = Vec::<f32>::try_from(&flat).unwrap();
        let request = Request {
            model: script.to_path_buf(),
            task: Task::UNet {
                foreground_channel: 0,
                softmax: false,
                boundary_channel: 1,
                pieces: Pieces::fixed(2048),
            },
        };
        let image = Image {
            width: w,
            height: h,
            channels: 1,
            data: &data,
        };
        let config = || EngineConfig {
            devices: DeviceSelection::Cpu,
            gpu_slots: 1,
            cores: 4,
        };
        let Response::UNet {
            probabilities,
            boundary,
        } = evanalyzer_ai::infer(&request, image, config).unwrap()
        else {
            panic!("not a U-Net answer");
        };
        let mut planes = probabilities;
        planes.extend(boundary.expect("a boundary plane"));
        let bytes: Vec<u8> = planes.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out.join(format!("{name}.engine.f32")), bytes).unwrap();
        println!("{name} engine: {w}x{h}");
    }
    // The package's own expected output, for the 256² input.
    let expected = Tensor::read_npy(package.join("test_output_0.npy")).unwrap();
    write_f32(&out.join("test256.expected.f32"), &expected);
}
