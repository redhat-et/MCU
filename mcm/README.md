# Model Cache Manager (MCM)

<img src="../logo/mcm.png" alt="mcm" width="20%" height="auto">

A lightweight CLI tool for **indexing, searching, and managing GPU kernel caches** for Triton and vLLM. MCM helps you organize, prune, and pre-warm caches to improve runtime efficiency and save disk space.

![MCM Screenshot](./screenshot/mcm-screenshot.png)

## Features

- **Cache Indexing** – Scan Triton or vLLM cache directories and build a local database recording kernel metadata
- **Advanced Search** – Query kernels by name, backend, architecture, modification time, and cache hit statistics
- **Intelligent Pruning** – Remove kernels based on age, usage patterns, or duplicates with multiple pruning strategies
- **Cache Warming** – Pre-generate vLLM caches in containers (CUDA/ROCm) and export
- **Runtime Tracking** – Monitor cache hits and access patterns in production environments

## Installation

### Prerequisites

- Python 3.9 or newer
- Triton
- Podman (required for `mcm warm` command - used to run vLLM containers)

### Install from Source

```bash
pip install -e .
```

### Verify Installation

```bash
# Check MCM is installed
mcm --help

# Verify Podman for cache warming (optional)
podman --version
```

## Quick Start

```bash
# Index your cache (auto-detects Triton or vLLM)
mcm index

# List all kernels
mcm list

# List CUDA kernels
mcm list --backend cuda

# Remove old unused kernels
mcm prune --older-than 30d --cache-hit-lower 5
```

## Usage

### Indexing the Cache

The `index` command scans your cache directory and populates the database with kernel metadata.

```bash
mcm index [OPTIONS]
```

**Options:**

- `--cache-dir PATH` – Path to cache directory (default: `~/.triton/cache` for Triton, `~/.cache/vllm` for vLLM)
- `--mode MODE` – Cache mode: `triton` or `vllm` (auto-detected if not specified)

**Examples:**

```bash
# Auto-detect and index
mcm index

# Index vLLM cache explicitly
mcm index --mode vllm --cache-dir ~/.cache/vllm

# Index custom Triton cache location
mcm index --cache-dir /path/to/my/cache
```

### Searching and Listing Kernels

The `list` command provides powerful search capabilities with multiple filter options.

```bash
mcm list [OPTIONS]
```

**Filter Options:**

- `--name, -n TEXT` – Filter by exact kernel name
- `--backend, -b TEXT` – Filter by backend (cuda, rocm, etc.)
- `--arch, -a TEXT` – Filter by architecture (80, 90a, gfx90a, etc.)
- `--older-than TEXT` – Show kernels older than duration (e.g., 7d, 2w)
- `--younger-than TEXT` – Show kernels younger than duration
- `--cache-hit-lower INT` – Show kernels with cache hits lower than specified number
- `--cache-hit-higher INT` – Show kernels with cache hits higher than specified number
- `--cache-dir PATH` – Specify cache directory to search
- `--mode MODE` – Cache mode (triton or vllm)

**Examples:**

```bash
# List all CUDA kernels
mcm list --backend cuda

# Find specific kernel by name
mcm list --name flash_attention_kernel

# Find kernels for specific GPU architecture
mcm list --backend cuda --arch 80

# Find old kernels with low usage
mcm list --older-than 30d --cache-hit-lower 10

# Find frequently used kernels
mcm list --cache-hit-higher 100

# Combine multiple filters
mcm list --backend rocm --arch gfx90a --younger-than 7d

# Search in vLLM cache
mcm list --mode vllm --backend cuda
```

The output displays a formatted table showing:

- Kernel hash (truncated for readability)
- Kernel name
- Cache hits count
- Last access time
- Backend and architecture
- Triton version
- Number of warps
- Total size
- Cache directory

### Pruning Kernels

The `prune` command removes kernels from the cache with the same powerful filtering as `list`, plus additional pruning strategies.

```bash
mcm prune [OPTIONS]
```

**Filter Options (same as list):**

- `--name, -n TEXT` – Target specific kernel name
- `--backend, -b TEXT` – Target specific backend
- `--arch, -a TEXT` – Target specific architecture
- `--older-than TEXT` – Delete kernels older than duration
- `--younger-than TEXT` – Delete kernels younger than duration
- `--cache-hit-lower INT` – Delete kernels with fewer hits
- `--cache-hit-higher INT` – Delete kernels with more hits
- `--cache-dir PATH` – Cache directory
- `--mode MODE` – Cache mode

**Pruning Options:**

- `--full` – Delete entire kernel directory (default: only remove IR files)
- `--deduplicate` – Keep only the newest copy of duplicate kernels
- `-y, --yes` – Skip confirmation prompt

**Pruning Strategies:**

1. **IR-only pruning (default)** – Removes intermediate representation files (.ttir, .ttgir, .llir) while keeping binaries
2. **Full pruning** – Removes entire kernel directory and database records
3. **Deduplication** – Identifies duplicate kernels and keeps only the newest version

**Examples:**

```bash
# Remove IR files from kernels older than 90 days
mcm prune --older-than 90d

# Fully remove all kernels with specific name
mcm prune --name unstable_kernel --full -y

# Remove unused old CUDA kernels
mcm prune --backend cuda --older-than 60d --cache-hit-lower 5 --full

# Deduplicate cache (removes older copies of duplicate kernels)
mcm prune --deduplicate -y

# Target specific architecture
mcm prune --arch gfx90a --older-than 30d

# Remove kernels in specific hit range
mcm prune --cache-hit-lower 10 --cache-hit-higher 50 --full

# Clean vLLM cache
mcm prune --mode vllm --older-than 14d
```

### Warming the Cache

The `warm` command pre-compiles GPU kernels for vLLM models by running actual inference in a containerized environment. This eliminates cold-start compilation delays in production deployments.

```bash
mcm warm [OPTIONS]
```

**How it works:**

1. Launches a vLLM container (CUDA or ROCm) with the specified model
2. Runs sample text generations to trigger kernel compilation
3. Collects environment metadata (GPU info, vLLM/Torch versions, Triton cache keys, etc)
4. Saves compiled kernels to the mounted cache directory
5. Optionally packages everything as a portable tarball

**Options:**

- `--model, -m TEXT` – Hugging Face model ID (default: facebook/opt-125m)
- `--output, -o PATH` – Output tarball path (default: warmed_cache.tar.gz)
- `--host-cache-dir PATH` – Host directory for cache (default: ./)
- `--hugging-face-token TEXT` – Token for private models
- `--vllm_cache_dir PATH` – Cache path inside container (default: /root/.cache/vllm/)
- `--tarball` – Create gzipped tarball after warming
- `--rocm` – Use ROCm image instead of CUDA

**What gets cached:**

- Compiled Triton kernels
- vLLM and Torch inductor compilation artifacts
- Metadata JSON

**Examples:**

```bash
# Basic cache warming
mcm warm --model facebook/opt-125m

# Create portable cache tarball for deployment
mcm warm --model meta-llama/Llama-2-7b-hf \
         --tarball \
         --output llama2_cache.tar.gz

# Warm cache for private model
mcm warm --model org/private-model \
         --hugging-face-token hf_xxxxx

# ROCm GPU support
mcm warm --model EleutherAI/gpt-neo-125M --rocm
```

The warm command uses container images:

- **CUDA**: `quay.io/rh-ee-asangior/vllm-0.9.2-tcm-warm:0.0.2` (based on vllm/vllm-openai)
- **ROCm**: `quay.io/rh-ee-asangior/vllm-0.9.1-tcm-warm-rocm:0.0.1` (based on rocm/vllm-dev)

**Generated Cache Contents:**

- Compiled Triton kernels in `torch_compile_cache/<hash>/rank<x>_<y>/triton_cache/`
- Metadata JSON with environment profile and cache keys
- All compilation artifacts from running inference on the model

## Database Structure

MCM uses SQLite databases to store kernel metadata:

- Triton mode: `~/.local/share/model-cache-manager/cache.db`
- vLLM mode: `~/.local/share/model-cache-manager/cache_vllm.db`

The database tracks:

- Kernel metadata (name, backend, architecture, version)
- File information (paths, sizes, types)
- Runtime statistics (hit counts, last access time)
- Kernel parameters (warps, stages, shared memory, etc.)

## Runtime Tracking

MCM includes a runtime tracker that can be integrated into your Triton code to monitor cache performance:

```python
from model_cache_manager.runtime.tracker import MCMTrackingCacheManager

# Use as drop-in replacement for Triton's CacheManager
triton.knobs.cache.manager_class = MCMTrackingCacheManager
```

This tracks cache hits and access patterns, updating the MCM database with runtime statistics.

## Advanced Usage

### Verbose Logging

Use `-v` flags for increased verbosity:

```bash
mcm -v index    # WARNING level
mcm -vv index   # INFO level
mcm -vvv index  # DEBUG level
```

## Project Structure

```text
mcm/
├── model_cache_manager/
│   ├── cli/           # CLI commands and helpers
│   ├── services/      # Business logic (index, search, prune, warm)
│   ├── data/          # Database and repository layers
│   ├── models/        # Data models (Kernel, SearchCriteria)
│   ├── plugins/       # Backend plugins (CUDA, ROCm)
│   ├── strategies/    # Mode strategies (Triton, vLLM)
│   ├── runtime/       # Runtime tracking
│   └── utils/         # Utilities and helpers
└── tests/             # Test suite
```

## Use Cases

### Development and Testing

- **Cache Analysis**: Understand what kernels your models generate (especially if autotune is involved)
- **Performance Debugging**: Track cache hit rates
- **Storage Management**: Keep development machines clean with intelligent pruning

### Production Deployments

- **Cache Pre-warming**: Eliminate cold-start compilation delays
- **Container Integration**: Ship pre-compiled caches with Docker/Kubernetes deployments using mcv after using mcm warm
- **Multi-node Consistency**: Ensure all nodes have identical optimized kernels
- **Version Migration**: Safely update vLLM/Triton versions with pre-validated caches
