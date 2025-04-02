# MMAPEAK - CUDA Matrix Multiply Performance Benchmark

MMAPEAK is a CUDA-based benchmarking tool designed to measure the peak performance of matrix multiplication operations across various data types and tensor core configurations on NVIDIA GPUs.

## Overview

This tool measures the throughput of NVIDIA's Tensor Core operations using different precision formats:
- 4-bit integer (int4)
- 4-bit floating point (FP4)
- 4-bit floating point with group scale (MXFP4 G32, NVFP4 G16)
- 6-bit floating point (FP6)
- 6-bit floating point with group scale (MXFP6 G32)
- 8-bit integer (INT8)
- 8-bit floating point (FP8)
- 8-bit floating point with group scale (MXFP8 G32)
- 16-bit floating point (FP16, BF16)
- 32-bit floating point (TF32)

## Building

### Using CMake (recommended)

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
./mmapeak [options]
```

### Options

- `-t <seconds>`: Set target time for benchmarks in seconds (default: 3.0)
- `-h, --help`: Show help message

## Example Output

```
Running benchmarks with target time: 10.0 seconds
mma_s4s4s32_8_8_32
run: 9294.6 ms 1189.6 T(fl)ops
mma_mxf4mxf4f32_16_8_64
not supported
mma_nvf4nvf4f32_16_8_64
not supported
mma_f4f4f16_16_8_32
not supported
mma_f4f4f32_16_8_32
not supported
mma_f6f6f16_16_8_32
not supported
mma_f6f6f32_16_8_32
not supported
mma_mxf6mxf6f32_16_8_32
not supported
mma_mxf8mxf8f32_16_8_32
not supported
mma_f8f8f16_16_8_32
run: 8330.3 ms 701.5 T(fl)ops
mma_f8f8f32_16_8_32
run: 9055.2 ms 351.0 T(fl)ops
mma_s8s8s32_16_16_16
run: 8798.9 ms 680.7 T(fl)ops
mma_s8s8s32_32_8_16
run: 8047.5 ms 680.9 T(fl)ops
mma_f16f16f16_16_16_16
run: 8993.3 ms 350.1 T(fl)ops
mma_f16f16f16_32_8_16
run: 9052.7 ms 351.7 T(fl)ops
mma_f16f16f32_16_16_16
run: 9162.0 ms 176.8 T(fl)ops
mma_f16f16f32_32_8_16
run: 9090.1 ms 176.9 T(fl)ops
mma_bf16bf16f32_16_16_16
run: 9117.0 ms 176.3 T(fl)ops
mma_bf16bf16f32_32_8_16
run: 9031.1 ms 176.3 T(fl)ops
mma_tf32tf32f32_16_16_8
run: 9128.7 ms 87.8 T(fl)ops
```

## Compatibility

The tool detects the available hardware capabilities at runtime. Tensor core operations that are not supported on your hardware will display "not supported".

## Architecture Support

- Ampere (A100, A30, etc.): SM80
- Ampere (A40, 3090, etc.): SM86
- Ada Lovelace (L40, 4090, etc.): SM89
- Hopper (H100, H200): SM90
- Blackwell (5090, etc.): SM120a

## License

This project is provided as-is.
