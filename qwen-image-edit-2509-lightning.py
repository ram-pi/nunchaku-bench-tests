import math
import time

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
from diffusers.utils import load_image

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

# From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

num_inference_steps = 4  # you can also use the 8-step model to improve the quality
rank = 32  # you can also use the rank=128 model to improve the quality

precision = get_precision()
print(f"Detected precision: {precision}")

model_path = f"nunchaku-tech/nunchaku-qwen-image-edit-2509/lightning-251115/svdq-{precision}_r{rank}-qwen-image-edit-2509-lightning-{num_inference_steps}steps-251115.safetensors"

# Measure model loading time
start_load = time.time()

# Load the model
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509", transformer=transformer, scheduler=scheduler, torch_dtype=torch.bfloat16
)

# Optimized offloading for RTX 4000 Ada (20GB VRAM)
# Use per-layer offloading with more blocks on GPU for better performance
gpu_mem = get_gpu_memory()
print(f"GPU Memory: {gpu_mem:.1f} GB")

if gpu_mem > 22:  # For GPUs with >22GB, try no offloading
    print("Loading to GPU (no offloading)...")
    pipeline.to("cuda")
elif gpu_mem > 18:  # For 20GB GPUs, use optimized per-layer offloading
    print("Using optimized per-layer offloading (4 blocks on GPU)...")
    transformer.set_offload(
        True, use_pin_memory=False, num_blocks_on_gpu=4  # Keep 4 blocks in GPU (faster than 1)
    )
    pipeline._exclude_from_cpu_offload.append("transformer")
    pipeline.enable_sequential_cpu_offload()
else:  # For <18GB GPUs, use aggressive offloading
    print("Using aggressive offloading (1 block on GPU)...")
    transformer.set_offload(
        True, use_pin_memory=False, num_blocks_on_gpu=1
    )
    pipeline._exclude_from_cpu_offload.append("transformer")
    pipeline.enable_sequential_cpu_offload()

torch.cuda.synchronize()  # Wait for all GPU operations to complete
end_load = time.time()
load_time = end_load - start_load
print(f"Model loading time: {load_time:.2f} seconds")

image1 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/man.png")
image1 = image1.convert("RGB")
image2 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/puppy.png")
image2 = image2.convert("RGB")
image3 = load_image("https://huggingface.co/datasets/nunchaku-tech/test-data/resolve/main/inputs/sofa.png")
image3 = image3.convert("RGB")

prompt = "Let the man in image 1 lie on the sofa in image 3, and let the puppy in image 2 lie on the floor to sleep."
inputs = {
    "image": [image1, image2, image3],
    "prompt": prompt,
    "true_cfg_scale": 1.0,
    "num_inference_steps": num_inference_steps,
}

# Measure execution time with CUDA events for accurate GPU timing
print("\nStarting inference with CUDA event timing...")

# Create CUDA events for precise GPU timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Record start event
start_event.record()

# Run inference
output = pipeline(**inputs)

# Record end event
end_event.record()

# Wait for GPU to finish all operations
torch.cuda.synchronize()

# Calculate elapsed time in milliseconds, then convert to seconds
cuda_time_ms = start_event.elapsed_time(end_event)
exec_time = cuda_time_ms / 1000.0

print(f"Execution time (CUDA): {exec_time:.3f} seconds")

# Save output
output_image = output.images[0]
output_image.save(f"/tmp/qwen-image-edit-2509-lightning-r{rank}-{num_inference_steps}steps.png")
print(f"Image saved as /tmp/qwen-image-edit-2509-lightning-r{rank}-{num_inference_steps}steps.png")
