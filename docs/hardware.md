# Local training and deployment budgets

The starting platform is the user's RTX 5070 Ti. NVIDIA specifies 16GB of memory for this GPU. Previously reported context is a Ryzen 7 7700X, 32GB system RAM, and native Windows; reconfirm current host details before sizing workloads. No desktop hardware has been measured in this session. [NVIDIA specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/).

## Why active parameters are not the memory budget

Inactive experts still have weights. During conventional full training, parameters generally also have optimizer state; experts inactive for one token can become active for another. Offloading moves a cost to RAM, storage, or interconnect bandwidth. It does not erase it.

Illustrative full-training accounting: 2 bytes of BF16 weights + 2 bytes of gradients + 4 bytes of master weights + 8 bytes of Adam moments = 16 bytes per trainable parameter. Actual implementations differ. This estimate excludes activations, temporary buffers, caches, fragmentation, and the display's memory use.

| Total trainable parameters | Illustrative parameter/state memory, decimal GB |
| --- | ---: |
| 100M | 1.6 |
| 300M | 4.8 |
| 1B | 16.0 |
| 8B | 128.0 |

This is why a 100–300M prototype is a reasonable profiling starting point while a multi-billion-parameter scratch-training promise would be premature. A fitted inference checkpoint does not imply a fitted training run. Adapter training has a different state budget; newly introduced modules still need training and storage.

## Deployment estimate, not a fit guarantee

At exactly four bits per parameter, raw weight storage is `parameters × 0.5` bytes:

| Total parameters | Raw Q4 weight floor, GiB |
| --- | ---: |
| 4B | 1.86 |
| 8B | 3.73 |
| 14B | 6.52 |
| 48B | 22.35 |

Quantization metadata, higher-precision modules, NCP state, memory tables, KV cache, and runtime buffers add to this floor. Thus a 48B model does not fit entirely on a 16GB card at Q4 just because 3B parameters activate per token. A 14B MoE that fits can still lose to a smaller dense model when routing and small matrix operations dominate.

For ordinary grouped-query attention, a useful KV-cache estimate is:

`2 × layers × batch × sequence_length × kv_heads × head_dim × bytes_per_element`

Custom concept modules or mixed local/global attention require their own accounting. Final deployment tests must name context length, batch/concurrency, quantization format, runtime, and host offload. An 8GB claim requires an actual 8GB configuration test.

## Desktop preflight

Run these read-only commands in native Windows PowerShell and retain their output with the first run record:

```powershell
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
Get-CimInstance Win32_Processor | Select-Object Name
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
python --version
python -m pip show torch
```

If PyTorch is already installed:

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda build', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('compiled architectures', torch.cuda.get_arch_list()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

These commands identify the environment; they are not a throughput benchmark. Missing Python or PyTorch should be recorded, not treated as a GPU failure. The CUDA version shown by nvidia-smi describes driver capability, not necessarily the installed training runtime.

Next, verify an actual GPU forward/backward/optimizer step on a supported PyTorch CUDA build. Start with standard attention operations before requiring optional compiled kernels. Upstream Linux recipes, old dependency pins, and advertised GPU support are not proof that our native Windows stack works.

## Cloud trigger

After measuring end-to-end training throughput, estimate `remaining training tokens / measured tokens per second`, then add evaluation, checkpointing, and startup time. Compare this with the declared local time budget. Move a validated experiment to cloud when it cannot fit locally after reasonable memory controls, or the projected duration exceeds that budget.

Carry the same model/data revisions, token budget, seeds, and run manifest to cloud. Re-profile there; do not extrapolate from GPU specifications. Select the provider, GPU, current price, storage/egress cost, and spending cap when an actual run is ready. No cloud resources are provisioned by this project.
