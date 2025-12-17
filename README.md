# Nunchaku Benchmark Tests

Benchmark scripts for testing Nunchaku-optimized Qwen-Image-Edit-2509 Lightning model with performance measurements using CUDA event timing.

## System Requirements

- **OS**: Ubuntu 24.04 (or compatible Linux distribution)
- **GPU**: NVIDIA GPU with CUDA support
  - Minimum: 8GB VRAM (with offloading)
  - Recommended: 20GB+ VRAM for better performance
- **Python**: 3.11 or higher
- **CUDA**: 11.8 or higher

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd nunchaku-bench-tests
```

### 2. Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### 3. Install Nunchaku and Dependencies

Follow **Option 1** from the [Nunchaku Installation Documentation](https://nunchaku.tech/docs/nunchaku/installation/installation.html):

```bash
# Step 1: Install PyTorch with CUDA support
# For CUDA 12.8 (adjust based on your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Step 2: Install Nunchaku from GitHub releases
# IMPORTANT: Choose the correct wheel for your setup:
# - Nunchaku version (e.g., v0.3.1)
# - PyTorch version (e.g., torch2.7)
# - Python version (e.g., cp311 for Python 3.11, cp312 for Python 3.12)
# - Platform (linux_x86_64 for Ubuntu)

# Example for Python 3.11 + PyTorch 2.7:
pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v0.3.1/nunchaku-0.3.1+torch2.7-cp311-cp311-linux_x86_64.whl

# Step 3: Install diffusers and other required packages
pip install diffusers transformers accelerate safetensors
```

**Finding the Right Wheel:**

1. Visit the [Nunchaku Releases Page](https://github.com/nunchaku-tech/nunchaku/releases)
2. Select the latest version (or specific version you want to test)
3. Find the wheel file matching your:
   - **Python version**: Check with `python --version`
     - Python 3.11 → `cp311`
     - Python 3.12 → `cp312`
   - **PyTorch version**: Check with `pip show torch`
     - PyTorch 2.7.x → `torch2.7`
     - PyTorch 2.9.x → `torch2.9`
   - **Platform**: `linux_x86_64` for Ubuntu 24.04

**Wheel Naming Convention:**
```
nunchaku-{version}+torch{pytorch_version}-cp{python_version}-cp{python_version}-{platform}.whl

Example:
nunchaku-0.3.1+torch2.7-cp311-cp311-linux_x86_64.whl
         ^^^    ^^^^^   ^^^^^ ^^^^^  ^^^^^^^^^^^^^^
          |        |       |     |          |
   nunchaku  pytorch  python python    platform
   version   version  version version
```

**Note**: Always check the [Nunchaku Installation Documentation](https://nunchaku.tech/docs/nunchaku/installation/installation.html) for the latest instructions and CUDA version compatibility.

### 4. Authenticate with Hugging Face

Some models require Hugging Face authentication. You can authenticate using either method:

**Method 1: Environment Variable (Recommended)**
```bash
export HF_TOKEN=your_huggingface_token_here
```

**Method 2: Hugging Face CLI**
```bash
# Install Hugging Face CLI
pip install -U huggingface_hub

# Login to Hugging Face
huggingface-cli login
```

Get your token from https://huggingface.co/settings/tokens

## Available Scripts

### `qwen-image-edit-2509-lightning.py`
Qwen-Image-Edit-2509 Lightning with adaptive offloading and single inference.
- **Features**:
  - Auto-detects GPU memory and adjusts offloading
  - CUDA event timing for precise measurements
  - Edits images using 3 input images
  - Saves single output to `/tmp/`
- **VRAM**: Adaptive (4-24GB)
- **Usage**:
  ```bash
  python qwen-image-edit-2509-lightning.py
  ```
- **Output**: `/tmp/qwen-image-edit-2509-lightning-r32-4steps.png`

### `qwen-image-edit-2509-lightning-10min.py`
10-minute benchmark saving all generated images.
- **Features**:
  - Runs for 10 minutes
  - Saves ALL generated images to `/tmp/`
  - CUDA event timing per iteration
  - Comprehensive statistics (avg, min, max times, throughput)
  - Adaptive offloading based on GPU memory
- **VRAM**: Adaptive (4-24GB)
- **Usage**:
  ```bash
  python qwen-image-edit-2509-lightning-10min.py
  ```
- **Output**: `/tmp/qwen-image-edit-2509-lightning-r32-4steps-iter001.png`, `iter002.png`, etc.

## Understanding the Output

### Timing Measurements

All scripts report timing using CUDA events for accurate GPU measurements:

```
Detected precision: int4
Model loading time: 45.23 seconds

Starting inference with CUDA event timing...
Execution time (CUDA): 5.234 seconds
```

- **Model loading time**: Time to load model from disk to GPU (CPU time with `torch.cuda.synchronize()`)
- **Execution time (CUDA)**: Pure GPU execution time using CUDA events (more accurate than `time.time()`)

### Precision Detection

The scripts auto-detect precision (`int4` or `fp4`) based on your GPU architecture:
- **fp4**: For Blackwell GPUs (RTX 50-series) - uses NVIDIA's native NVFP4 format for better quality
- **int4**: For Ada Lovelace, Ampere, and Turing architectures

### Offloading Strategies

Scripts use different offloading strategies based on available VRAM:

| VRAM | Strategy | Performance | Applied by Scripts |
|------|----------|-------------|--------------------|
| <18GB | Aggressive offloading (1 block on GPU) | Slower, very low VRAM | Both scripts (auto-detected) |
| 18-24GB | Optimized offloading (4 blocks on GPU) | Balanced | Both scripts (auto-detected) |
| >24GB | No offloading (entire model on GPU) | Fastest | Both scripts (auto-detected) |

## Performance Tips

### For Low VRAM (8-16GB)
- Scripts automatically use aggressive offloading (1 block on GPU)
- Expect slower inference times (~10-15s per image)
- Monitor VRAM usage: `watch -n 1 nvidia-smi`

### For Medium VRAM (20-24GB)
- Scripts automatically use optimized offloading (4 blocks on GPU)
- Balanced performance (~5-8s per image)
- Can manually adjust `num_blocks_on_gpu` in scripts for tuning

### For High VRAM (40GB+)
- Scripts automatically disable offloading for maximum speed
- Best performance (~3-5s per image)
- Use 10-minute benchmark script to test sustained performance

## Troubleshooting

### CUDA Out of Memory Error

```
torch.OutOfMemoryError: CUDA out of memory
```

**Solutions**:
1. Scripts automatically adjust offloading based on VRAM, but you can manually reduce `num_blocks_on_gpu` in the script (change from 4 to 2 or 1)
2. Close other GPU-using applications
3. Restart the script (sometimes helps clear cached memory)

### Slow First Run

The first run is always slower due to:
- Model download from Hugging Face Hub
- CUDA kernel initialization
- Python/PyTorch JIT compilation

Subsequent runs will be much faster as models are cached locally.

## Monitoring GPU Usage

Monitor GPU usage in real-time:

```bash
# Watch GPU memory and utilization
watch -n 1 nvidia-smi

# Or use a more detailed view
nvidia-smi dmon -s mu
```

## Resources

- [Nunchaku Documentation](https://nunchaku.tech/docs/)
- [Nunchaku Installation Guide](https://nunchaku.tech/docs/nunchaku/installation/installation.html)
- [Hugging Face Model Hub](https://huggingface.co/nunchaku-tech)
