# vLLM Binary Cache Support

## Overview

MCV supports three vLLM cache formats:

1. **vLLM Triton Cache Format** (legacy) - Stores `triton_cache/` and
   `inductor_cache/` inside rank directories
2. **vLLM Binary Cache Format** (default) - Stores compiled artifacts in prefix
   directories with embedded Triton kernels
3. **vLLM AOT Cache Format** (advanced) - Uses `VLLM_USE_MEGA_AOT_ARTIFACT=true`
   for fully self-contained portable artifacts

All formats share the same top-level structure:
`torch_compile_cache/{hash}/rank_{rank}_{dp_rank}/`

The key differences are **inside the rank directory**:

- **Triton format**: Contains `triton_cache/` and `inductor_cache/`
  subdirectories with unpacked artifacts
- **Binary format**: Contains prefix directories
  (e.g., `backbone/`, `eagle_head/`) with `cache_key_factors.json`
  and binary artifacts containing embedded Triton kernels
- **AOT format**: Identical structure to binary format, but uses PyTorch's
  `AOTCompiledArtifact` serialization (indicated by
  `VLLM_USE_MEGA_AOT_ARTIFACT: true` in `cache_key_factors.json`)

This document describes the **vLLM Binary and AOT Cache Formats** and how
torch.compile caching works with MCV.

## Torch Compile Architecture

### How vLLM Uses torch.compile

When vLLM is configured with `VLLM_TORCH_COMPILE_LEVEL=1`, it uses PyTorch's
`torch.compile` with TorchInductor backend to optimize model execution:

```text
Model Code → torch.compile → TorchInductor → Triton/CUDA Kernels → GPU Execution
```

**First Run (Compilation)**:

1. vLLM traces the model with Dynamo
2. TorchInductor compiles the graph
3. Triton generates optimized GPU kernels → `/tmp/torchinductor_root/`
4. vLLM saves artifacts using `standalone_compile().save(format="binary")`
5. **PyTorch bundles the Triton kernels into the artifacts**
6. Complete cache saved to `~/.cache/vllm/torch_compile_cache/`

**Subsequent Runs (Cache Hit)**:

1. vLLM loads artifacts from `~/.cache/vllm/torch_compile_cache/`
2. **PyTorch extracts embedded Triton kernels → `/tmp/torchinductor_root/`**
3. Execution resumes using extracted kernels (~10-20s vs 3-5min compilation)

### Binary vs AOT Formats

Both binary and AOT formats bundle Triton kernels in the artifacts, but differ
in serialization:

**Binary Format** (default):

- Uses PyTorch `standalone_compile().save(format="binary")`
- Environment: `VLLM_USE_MEGA_AOT_ARTIFACT=false` (default)
- Good for same PyTorch version deployments
- Typical size: ~95MB for small models

**AOT Format** (advanced):

- Uses PyTorch `AOTCompiledArtifact.serialize()`
- Environment: `VLLM_USE_MEGA_AOT_ARTIFACT=true`
- More portable across PyTorch versions (requires 2.10+)
- Includes bundled AOT autograd cache
- Typical size: ~92MB for small models

**Important**: From MCV's perspective, both formats are **structurally identical**
and use the same detection and packaging logic.

### The /tmp Cache Directory

During compilation and execution, PyTorch creates temporary files:

```text
/tmp/torchinductor_root/
├── triton/0/{hash}/
│   ├── triton_.cubin    # Compiled GPU binary (ELF)
│   ├── triton_.source   # Triton source code
│   ├── triton_.ttir     # Triton IR
│   └── triton_.ptx      # PTX assembly
├── o7/, dp/, .../       # Python kernel cache
└── aotautograd/         # AOT autograd cache
```

**Size**: ~16MB for small models

**Lifecycle**:

- **First run**: Created during compilation
- **Cache hit**: Extracted from embedded artifacts
- **Cleanup**: Cleared on reboot (tmpfs) or manual deletion
- **Recreation**: Automatic on every vLLM start

**Key Insight**: This directory is **NOT needed for cache portability**.
The Triton kernels are already embedded in the binary artifacts (verified by
finding 42 ELF headers in a 5.3MB artifact file).

**MCV does NOT capture `/tmp`** - kernels auto-extract at runtime (~2 seconds).

## Binary Cache Format

### Directory Structure

The binary cache uses a structured directory layout:

```text
torch_compile_cache/
└── {hash}/                           # 10-character cache hash
    └── rank_{rank}_{dp_rank}/        # Per-rank cache
        └── {prefix}/                 # Model component
            ├── cache_key_factors.json
            ├── vllm_compile_cache.py
            ├── computation_graph.py
            └── artifact_compile_range_{start}_{end}_subgraph_{i}
```

### Key Components

#### 1. Cache Hash Directory

The hash directory is a 10-character truncated SHA256 hash derived from:

- Code hash (SHA256 of forward code files)
- Configuration hash (hash of vLLM config)
- Compiler hash (Inductor compiler state)
- Environment hash (compilation-affecting env vars)

#### 2. Rank Directory

Format: `rank_{rank}_{dp_rank}`

- `{rank}`: Distributed training rank
- `{dp_rank}`: Data parallel rank
- Allows multiple ranks to maintain separate caches

#### 3. Prefix Directory

Common prefixes:

- `backbone`: Main model component (default)
- `eagle_head`: Speculative decoding draft model

#### 4. Cache Files

**cache_key_factors.json**: Metadata tracking cache key components

```json
{
  "code_hash": "<sha256-hash>",
  "compiler_hash": "<compiler-hash>",
  "config_hash": "<config-hash>",
  "env": {
    "VLLM_TARGET_DEVICE": "cuda",
    "VLLM_COMPILE_CACHE_SAVE_FORMAT": "binary",
    "VLLM_MAIN_CUDA_VERSION": "12.9",
    ...
  }
}
```

**vllm_compile_cache.py**: Python dict mapping compile ranges to artifact
handles

**computation_graph.py**: Readable FX graph source code (for debugging)

**artifact_compile_range_* files**: Compiled artifacts

- **Binary format** (default): Single binary file per artifact
- **Unpacked format**: Directory containing Inductor output files

## Storage Formats

vLLM supports two storage formats for artifacts, controlled by
`VLLM_COMPILE_CACHE_SAVE_FORMAT`:

### Binary Format (default)

- **Env Var**: `VLLM_COMPILE_CACHE_SAVE_FORMAT=binary`
- **Artifacts**: Regular files
- **Multiprocess Safe**: Yes
- **Inspection**: Cannot easily inspect contents
- **Use Case**: Production deployments

```text
{prefix}/
├── artifact_compile_range_{start}_{end}_subgraph_0  (file, ~2.7 MB)
└── artifact_compile_range_{start}_{end}_subgraph_1  (file, ~2.1 MB)
```

### Unpacked Format

- **Env Var**: `VLLM_COMPILE_CACHE_SAVE_FORMAT=unpacked`
- **Artifacts**: Directories with Python/Triton files
- **Multiprocess Safe**: No (race conditions possible)
- **Inspection**: Can view and debug generated code
- **Use Case**: Development and debugging

```text
{prefix}/
├── artifact_compile_range_{start}_{end}_subgraph_0/  (directory)
│   ├── kernel_0.py
│   └── kernel_1.py
└── artifact_compile_range_{start}_{end}_subgraph_1/  (directory)
```

## MCV Metadata

### Container Image Labels

When a binary cache is packaged in a container image, MCV adds the
following labels:

```json
{
  "cache.vllm.image/entry-count": "1",
  "cache.vllm.image/cache-size-bytes": "35702329",
  "cache.vllm.image/format": "binary",
  "cache.vllm.image/summary": "{\"targets\":[...]}"
}
```

**Label Descriptions:**

- `entry-count`: Number of cache hash directories detected
- `cache-size-bytes`: Total size of the cache in bytes
- `format`: Storage format (`"binary"` or `"unpacked"`)
- `summary`: Hardware target information (JSON)

### Manifest Structure

The `manifest.json` file contains comprehensive metadata:

```json
{
  "vllm": [
    {
      "vllmHash": "{hash}",
      "cacheFormat": "binary",
      "binary": [
        {
          "rank": "rank_{rank}_{dp_rank}",
          "prefix": "{prefix}",
          "artifact_count": 17,
          "artifact_names": [
            "artifact_compile_range_{start}_{end}_subgraph_0",
            "artifact_compile_range_{start}_{end}_subgraph_1",
            ...
          ],
          "code_hash": "<sha256-hash>",
          "config_hash": "<config-hash>",
          "compiler_hash": "<compiler-hash>",
          "cache_save_format": "binary",
          "target_device": "cuda",
          "env": {
            "VLLM_TARGET_DEVICE": "cuda",
            "VLLM_COMPILE_CACHE_SAVE_FORMAT": "binary",
            "VLLM_MAIN_CUDA_VERSION": "12.9",
            ...
          }
        }
      ]
    }
  ]
}
```

**Manifest Fields:**

- `cacheFormat`: vLLM cache structure type (`"binary"` for new binary cache
  format, `"triton"` for legacy triton cache format)
- `binary[]`: Array of binary cache entries (one per rank/prefix combination)
- `cache_save_format`: Actual artifact storage format (`"binary"` or
  `"unpacked"`)
- `target_device`: Target hardware (`"cuda"`, `"rocm"`, `"tpu"`, `"cpu"`)
- `env`: Full environment variables from `cache_key_factors.json`

## Hardware Detection

MCV automatically extracts hardware information from the cache metadata:

### CUDA

```json
{
  "backend": "cuda",
  "arch": "sm_12.9",
  "warp_size": 32
}
```

- **Backend**: Extracted from `VLLM_TARGET_DEVICE`
- **Arch**: Derived from `VLLM_MAIN_CUDA_VERSION`
- **Warp Size**: 32 (CUDA default)

### ROCm/HIP

```json
{
  "backend": "rocm",
  "arch": "gfx90a",
  "warp_size": 64
}
```

- **Backend**: Extracted from `VLLM_TARGET_DEVICE`
- **Arch**: Detected from ROCm environment variables
- **Warp Size**: 64 (AMD wavefront size)

## Format Detection

MCV automatically detects the vLLM cache format by inspecting the
filesystem:

1. **vLLM Binary Cache Detection**:
   - Looks for `rank_X_Y/` directories
   - Checks for `cache_key_factors.json`
   - Inspects `artifact_compile_range_*` entries
   - If entries are **files** → Binary artifact storage
   - If entries are **directories** → Unpacked artifact storage

2. **vLLM Triton Cache Detection** (fallback):
   - Looks for `triton_cache/` directory
   - Uses legacy vLLM triton cache extraction logic

This filesystem-based detection is more reliable than environment variables,
especially when caches are copied between systems.

### Format Indicators

MCV uses **three distinct format indicators** to describe vLLM caches. Each
serves a different purpose:

#### 1. Manifest `cacheFormat` (Cache Structure Type)

**Location**: `manifest.json` → `vllm[].cacheFormat`

**Values**: `"binary"` or `"triton"`

**Purpose**: Tells MCV extraction logic which vLLM cache structure to expect
inside rank directories

- `"binary"`: vLLM binary cache format - rank directories contain prefix
  subdirectories (e.g., `backbone/`)
- `"triton"`: vLLM triton cache format - rank directories contain
  `triton_cache/` subdirectory

**Example**:

```json
{
  "vllm": [{
    "cacheFormat": "binary",  // ← Extraction logic uses this
    "binary": [...]
  }]
}
```

This field determines which extraction code path MCV uses and is essential
for correctly unpacking the cache from the container image.

#### 2. Manifest `cache_save_format` (Artifact Storage Format)

**Location**: `manifest.json` → `vllm[].binary[].cache_save_format`

**Values**: `"binary"` or `"unpacked"`

**Purpose**: Records the actual artifact storage format detected from the
filesystem

- `"binary"`: Artifacts are individual files (multiprocess-safe, production
  use)
- `"unpacked"`: Artifacts are directories containing Python/Triton source
  files (debugging use)

**Example**:

```json
{
  "vllm": [{
    "cacheFormat": "binary",
    "binary": [{
      "rank": "rank_0_0",
      "prefix": "backbone",
      "cache_save_format": "binary",  // ← Detected from filesystem
      "artifact_count": 17,
      ...
    }]
  }]
}
```

This field is informational and helps users understand the internal artifact
format.

#### 3. Image Label `format` (User-Visible Format)

**Location**: OCI image labels → `cache.vllm.image/format`

**Values**: `"binary"` or `"unpacked"`

**Purpose**: Quick user-visible indicator of artifact storage format

- `"binary"`: For vLLM binary cache format with binary artifacts
- `"unpacked"`: For vLLM triton cache format OR vLLM binary cache format with
  unpacked artifacts

**Example**:

```json
{
  "cache.vllm.image/format": "binary",  // ← Quick indicator for users
  "cache.vllm.image/entry-count": "1",
  "cache.vllm.image/cache-size-bytes": "35702329"
}
```

This label allows users to quickly inspect cache format using `docker
inspect` or `skopeo inspect` without reading the full manifest.

### Format Mapping Table

| vLLM Format | Artifacts | `cacheFormat` | `cache_save_format` | Label |
| ----------- | --------- | ------------- | ------------------- | ----- |
| Binary | Binary files | `"binary"` | `"binary"` | `"binary"` |
| Triton | Unpacked dirs | `"triton"` | N/A | `"unpacked"` |

**Why Three Indicators?**

- **Manifest `cacheFormat`**: Extraction logic must know what's inside rank
  directories (`triton_cache/` subdirs vs `{prefix}/` subdirs)
- **Manifest `cache_save_format`**: Detailed metadata for debugging and
  compatibility checking
- **Image Label `format`**: Fast user-facing indicator without parsing full
  manifest

## Comparison: vLLM Binary Cache vs vLLM Triton Cache

| Aspect | Triton (Legacy) | Binary (New) |
| ------ | --------------- | ------------ |
| **Structure** | `{hash}/rank_X_Y/` | `{hash}/rank_X_Y/` |
| **Inside Rank** | `triton_cache/` + `inductor_cache/` | `{prefix}/` |
| **Metadata** | Triton JSON | `cache_key_factors.json` |
| **Storage** | Unpacked | Binary/unpacked |
| **Multiprocess** | No | Yes (binary) |
| **Distributed** | Full rank/DP | Full rank/DP |
| **Manifest** | `"triton"` | `"binary"` |
| **Label** | `"unpacked"` | `"binary"`/`"unpacked"` |

## Usage Examples

### Building a Cache Image

```bash
# Build from binary cache directory
mcv -c -d /path/to/model-binary-cache \
    -i quay.io/myorg/model-cache:v1 \
    --builder docker

# Result includes labels and manifest
```

### Extracting a Cache Image

```bash
# Extract cache from image
mcv -e -i quay.io/myorg/model-cache:v1

# MCV automatically detects format from manifest
# and extracts to appropriate location
```

### Inspecting Cache Metadata

```bash
# View image labels
skopeo inspect docker://quay.io/myorg/model-cache:v1 \
  | jq '.Labels'

# Expected output:
# {
#   "cache.vllm.image/format": "binary",
#   "cache.vllm.image/summary": "{\"targets\":[...]}",
#   ...
# }
```

## vLLM Source References

Key files in vLLM that implement binary cache:

- `vllm/envs.py:1512-1520` - `VLLM_COMPILE_CACHE_SAVE_FORMAT` definition
- `vllm/compilation/compiler_interface.py:186-327` -
  `InductorStandaloneAdaptor`
- `vllm/compilation/backends.py:245-346` - Compilation manager
- `vllm/compilation/backends.py:904-935` - `cache_key_factors.json` creation
- `vllm/compilation/backends.py:867-874` - Directory structure creation

## Best Practices

1. **Use binary format in production** for multiprocess safety
2. **Use unpacked format for debugging** to inspect generated code
3. **Include full env in manifest** for cache compatibility checking
4. **Verify hardware match** using image labels before deployment
5. **Check cache_save_format** in manifest when extracting caches

## Migration from vLLM Triton Cache to vLLM Binary Cache

To migrate from vLLM triton cache format to vLLM binary cache format:

1. Update vLLM to a version that supports binary cache format
2. Set `VLLM_COMPILE_CACHE_SAVE_FORMAT=binary`
3. Run model warmup to generate new binary cache
4. Package new cache with MCV (automatically detected)
5. Both vLLM cache formats are supported, no breaking changes

## Practical Guide

### Generating a Cache

**Environment Setup**:

```bash
export VLLM_TORCH_COMPILE_MODE=vllm-compile
export VLLM_TORCH_COMPILE_LEVEL=1

# For binary format (default):
export VLLM_COMPILE_CACHE_SAVE_FORMAT=binary
export VLLM_USE_MEGA_AOT_ARTIFACT=false  # or omit (default)

# For AOT format (more portable):
export VLLM_COMPILE_CACHE_SAVE_FORMAT=binary
export VLLM_USE_MEGA_AOT_ARTIFACT=true  # requires PyTorch 2.10+
```

**Run vLLM Warmup**:

```bash
vllm serve my-model --tensor-parallel-size 1

# Make sample requests to trigger compilation:
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model", "prompt": "Hello", "max_tokens": 100}'
```

**Verify Cache**:

```bash
ls -lh ~/.cache/vllm/torch_compile_cache/
# Should show a 10-char hash directory (e.g., 8d0a361fbc)

# Check cache contents:
find ~/.cache/vllm/torch_compile_cache/ -type f | head
```

### Packaging with MCV

**Create Container Image**:

```bash
mcv -c \
  -d ~/.cache/vllm/torch_compile_cache/{hash} \
  -i quay.io/myorg/my-model-cache:v1
```

**Verify Image Labels**:

```bash
skopeo inspect containers-storage:quay.io/myorg/my-model-cache:v1 \
  | jq '.Labels'

# Expected labels:
# {
#   "cache.vllm.image/cache-size-bytes": "95000000",
#   "cache.vllm.image/entry-count": "1",
#   "cache.vllm.image/format": "binary",
#   "cache.vllm.image/summary": "{\"targets\":[{\"backend\":\"cuda\",...}]}"
# }
```

### Using a Cached Image

**Extract Cache**:

```bash
mcv -e -i quay.io/myorg/my-model-cache:v1

# MCV extracts to: ~/.cache/vllm/torch_compile_cache/{hash}/
```

**Start vLLM**:

```bash
# vLLM automatically detects and uses the cache
vllm serve my-model --tensor-parallel-size 1

# Look for log message:
# INFO: Directly load the compiled graph(s) from the cache, took X.X s
```

### Cache Compatibility

A cache is compatible if:

1. **GPU architecture** matches (check: `nvidia-smi --query-gpu=compute_cap`)
2. **CUDA/ROCm version** compatible (check: `nvcc --version` or `rocm-smi`)
3. **PyTorch version** compatible
4. **Model code** unchanged (code hash must match)
5. **vLLM configuration** matches (TP size, compile level, etc.)

**Check Compatibility**:

```bash
# View cache metadata:
cat ~/.cache/vllm/torch_compile_cache/*/rank_0_0/*/cache_key_factors.json \
  | jq '{target: .env.VLLM_TARGET_DEVICE, cuda: .env.VLLM_MAIN_CUDA_VERSION}'

# Compare with system:
nvidia-smi
# or
rocm-smi
```

## Troubleshooting

### Cache Not Being Used

**Symptom**: vLLM recompiles on every start despite having a cache

**Common Causes**:

1. **Hash mismatch** - Configuration or environment changed
2. **Incompatible GPU** - Different architecture (e.g., sm_75 vs sm_80)
3. **PyTorch version** - Binary format sensitive to PyTorch version
4. **Model code changed** - Code hash no longer matches

**Debug Steps**:

```bash
# 1. Check if cache exists
ls ~/.cache/vllm/torch_compile_cache/

# 2. Enable debug logging
export VLLM_LOGGING_LEVEL=DEBUG

# 3. Check for hash mismatch in logs
grep "cache" vllm.log | grep -i "hash\|miss"

# 4. Verify GPU compatibility
python -c "import torch; print(torch.cuda.get_device_capability())"
```

### Slow Startup with Cache

**Symptom**: vLLM takes 20+ seconds to start with cache

**Normal Behavior**: 10-20 seconds for kernel extraction from artifacts is expected

**If Slower**:

- Check disk I/O performance: `iostat -x 1`
- Verify `/tmp` is not on slow storage (NFS, etc.)
- Consider using `tmpfs` for `/tmp`: `df -h /tmp`

### Missing Kernels Error

**Symptom**: Runtime errors about missing Triton kernels

**Causes**:

1. Corrupted artifacts
2. Incomplete cache (warmup didn't cover all batch sizes)
3. Disk space issues during generation

**Solutions**:

```bash
# 1. Delete and regenerate cache
rm -rf ~/.cache/vllm/torch_compile_cache/*

# 2. Verify disk space
df -h ~/.cache/vllm/

# 3. Check artifact integrity
file ~/.cache/vllm/torch_compile_cache/*/rank_0_0/*/artifact_*
# Should show: "data" (binary format)
```

### AOT Format Issues

**Symptom**: AOT artifacts fail to load

**Requirements**:

- PyTorch 2.10.0 or later
- `VLLM_USE_MEGA_AOT_ARTIFACT=true`
- `VLLM_USE_STANDALONE_COMPILE=true`

**Verify**:

```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Verify AOT flag in cache
grep "VLLM_USE_MEGA_AOT_ARTIFACT" \
  ~/.cache/vllm/torch_compile_cache/*/rank_0_0/*/cache_key_factors.json
```

## Advanced Topics

### Multi-GPU Caching

For tensor parallelism or pipeline parallelism:

```text
torch_compile_cache/{hash}/
├── rank_0_0/    # First tensor parallel rank
├── rank_0_1/    # Second tensor parallel rank
├── rank_1_0/    # First pipeline parallel rank
└── rank_1_1/    # Second pipeline + tensor parallel rank
```

MCV captures all rank directories. Extract the entire hash directory for
multi-GPU deployments.

### Multiple Model Components

Models with speculative decoding have multiple components:

```text
rank_0_0/
├── backbone/        # Main model
│   └── artifact_*
└── eagle_head/      # Draft model for speculation
    └── artifact_*
```

MCV captures all prefix directories automatically.

### Cache Size Optimization

**Typical Sizes**:

- Small models (< 1B params): 50-100 MB
- Medium models (1-10B params): 100-500 MB
- Large models (10B+ params): 500 MB - 2 GB

**Factors Affecting Size**:

- Number of compiled ranges (batch sizes)
- Number of layers
- Triton kernel count
- Autotune configurations

**Reduce Size**:

- Use fewer compile ranges: `VLLM_COMPILE_RANGES=[128,512]` vs default
- Binary format is smaller than unpacked
- AOT format is similar to binary

## See Also

- [spec-compat.md](./spec-compat.md) - OCI image specification
- [design.md](./design.md) - MCV architecture and design
- [vLLM Documentation](https://github.com/vllm-project/vllm) - vLLM project
