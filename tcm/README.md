# Triton Cache Manager (TCM)

A lightweight CLI for **indexing, searching, and managing Triton GPU‑kernel caches**. TCM helps you organize, prune, and even pre‑warm your Triton kernel cache for improved efficiency and disk‑space management.

---

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Usage](#usage)

  * [Indexing the Cache](#indexing-the-cache)
  * [Listing Kernels (Search)](#listing-kernels-search)
  * [Pruning Kernels](#pruning-kernels)
  * [Warming the Cache](#warming-the-cache)
* [Requirements](#requirements)
* [Project Structure](#project-structure)

---

## Features

TCM provides the following key functionalities:

* **Cache Indexing:** Scans your Triton kernel cache directory to build and update a local database with detailed kernel metadata, including name, backend, architecture, and file sizes.
* **Flexible Search and Listing:** Query indexed kernels based on various criteria such as name, backend (for example *CUDA* or *ROCm*), architecture, and modification time (older or younger than specific durations).
* **Intelligent Pruning:**

  * **IR‑Only Pruning:** Delete only the intermediate representation (IR) files, preserving compiled binaries and metadata. This saves space while retaining functionality.
  * **Full Kernel Deletion:** Remove entire kernel directories from the cache along with their database entries.
  * **Deduplication:** Automatically detect and delete older duplicate kernel instances, keeping only the newest version of each unique kernel.
* **Cache Warming:** Pre‑fills the vLLM cache for chosen models in a containerised environment. Both *CUDA* and *ROCm* are supported, and the warmed cache can be packaged as a tarball for distribution.
* **Human‑Readable Output:** Presents kernel information in well‑formatted tables with readable file sizes and modification times.

---

## Installation

### Prerequisites

* Python 3.9 or higher
* Triton
* [Podman](https://podman.io/) (required for the `tcm warm` command)

### Installation Steps

1. **Install in editable mode**

   ```bash
   pip install -e .
   ```

   The command installs TCM and its dependencies (`typer`, `rich`, `sqlalchemy`, `pydantic`) and makes the `tcm` CLI available.

---

## Quick Start

1. **Index your Triton kernel cache**

   ```bash
   tcm index --cache-dir ~/.triton/cache   # or your custom cache path
   ```

2. **List kernels by backend**

   ```bash
   tcm list --backend cuda
   ```

---

## Usage

The `tcm` command‑line interface provides several sub‑commands.

### Indexing the Cache

The `index` command scans the specified cache directory (default `~/.triton/cache`) and populates the local database with kernel metadata.

```bash
tcm index [OPTIONS]
```

**Options**

* `--cache-dir PATH`  Specify the Triton cache directory (defaults to `~/.triton/cache` on Linux).

**Example**

```bash
tcm index --cache-dir /path/to/my/triton/cache
```

### Listing Kernels (Search)

Search and display indexed kernels based on various criteria.

```bash
tcm list [OPTIONS]
```

**Options**

* `--name -n TEXT`         Filter by kernel name (exact match)
* `--backend -b TEXT`      Filter by backend (e.g. `cuda`, `rocm`)
* `--arch -a TEXT`         Filter by architecture (e.g. `80`, `gfx90a`)
* `--older-than TEXT`      Show kernels older than a duration (e.g. `7d`)
* `--younger-than TEXT`    Show kernels younger than a duration (e.g. `14d`)
* `--cache-dir PATH`       Cache directory to search (uses default if omitted)

**Examples**

* List all CUDA kernels

  ```bash
  tcm list --backend cuda
  ```

* Find kernels named `my_custom_kernel` older than 30 days

  ```bash
  tcm list --name my_custom_kernel --older-than 30d
  ```

* List ROCm kernels for a specific architecture

  ```bash
  tcm list --backend rocm --arch gfx90a
  ```

### Pruning Kernels

Remove kernel files from the cache based on filters. Supports partial (IR‑only) or full deletions, and deduplication.

```bash
tcm prune [OPTIONS]
```

**Options**

* `--name -n TEXT`        Filter by kernel name
* `--backend -b TEXT`     Filter by backend
* `--arch -a TEXT`        Filter by architecture
* `--older-than TEXT`     Prune kernels older than the duration
* `--younger-than TEXT`   Prune kernels younger than the duration
* `--full`                Delete the entire kernel directory
* `--deduplicate`         Remove older duplicates (ignores other filters)
* `-y, --yes`             Skip the confirmation prompt
* `--cache-dir PATH`      Specify the cache directory

**Examples**

* Prune IR files of all kernels older than 90 days

  ```bash
  tcm prune --older-than 90d
  ```

* Fully remove all `unstable_kernel` entries

  ```bash
  tcm prune --name unstable_kernel --full -y
  ```

* Deduplicate kernels, keeping only the newest copies

  ```bash
  tcm prune --deduplicate -y
  ```

### Warming the Cache

Use Podman to warm the vLLM cache for a model and optionally package the result.

```bash
tcm warm [OPTIONS]
```

**Options**

* `--model -m TEXT`          Model on Hugging Face (default: `facebook/opt-125m`)
* `--output -o PATH`         Output file for the tarball (default: `warmed_cache.tar.gz`)
* `--host-cache-dir PATH`    Host directory for the vLLM cache (default: `./`)
* `--hugging-face-token TEXT`  Token for private models
* `--vllm_cache_dir PATH`    Cache directory *inside* the container (default: `/root/.cache/vllm/`)
* `--tarball`                Create a gzipped tarball of the warmed cache
* `--rocm`                   Warm for ROCm GPUs (default is CUDA)

**Examples**

* Warm cache for a model and create a tarball

  ```bash
  tcm warm --model EleutherAI/gpt-neo-125M \
           --tarball \
           --output my_gpt_neo_cache.tar.gz
  ```

* Warm cache for a ROCm system

  ```bash
  tcm warm --model Llama-3-8B --rocm \
           --hugging-face-token hf_YOUR_TOKEN
  ```

---

## Requirements

All core dependencies are listed in `requirements.txt` and are installed automatically:

* `typer`
* `rich`
* `pydantic`
* `pydantic-settings`
* `structlog`
* `sqlalchemy`

---

## Project Structure

The source code is organised into logical modules:

* `tcm/triton_cache_manager/cli`
  Command‑line interface entry points
* `tcm/triton_cache_manager/services`
  Core business logic for indexing, searching, pruning, and warming
* `tcm/triton_cache_manager/data`
  Data access layer (SQLite + SQLAlchemy) and cache repository operations
* `tcm/triton_cache_manager/models`
  DTOs and Pydantic models for kernel metadata and search criteria
* `tcm/triton_cache_manager/plugins`
  Extensible backend support (CUDA, ROCm) and related file handlers
* `tcm/triton_cache_manager/utils`
  Utility functions for logging, path management, and formatting
