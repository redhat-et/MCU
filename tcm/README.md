# Triton Cache Manager (TCM)

<img src="../logo/tcm.png" alt="tcm" width="20%" height="auto">

A lightweight CLI for **indexing, searching, and managing Triton GPU-kernel caches**. TCM helps you organize, prune, and even pre-warm your Triton kernel cache for improved efficiency and disk space management.

---

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Usage](#usage)
    * [Indexing the Cache](#indexing-the-cache)
    * [Listing Kernels (Search)](#listing-kernels-search)
    * [Pruning Kernels](#pruning-kernels)
    * [Warming the Cache](#warming-the-cache)
* [Requirements](#requirements)
* [Project Structure](#project-structure)

---

## Features

TCM provides the following key functionalities:

* **Cache Indexing:** Scans your Triton kernel cache directory to build and update a local database with detailed kernel metadata, including name, backend, architecture, and file sizes.
* **Flexible Search and Listing:** Query indexed kernels based on various criteria such as name, backend (e.g., CUDA, ROCm), architecture, and modification time (older/younger than specific durations).
* **Intelligent Pruning:**
    * **IR-Only Pruning:** Delete only the intermediate representation (IR) files for kernels, preserving compiled binaries and metadata to save space while retaining functionality.
    * **Full Kernel Deletion:** Remove entire kernel directories from the cache and corresponding database entries.
    * **Deduplication:** Automatically identify and delete older duplicate kernel instances, keeping only the newest version of each unique kernel.
* **Cache Warming:** Pre-fills the vLLM cache for specified models using a containerized environment, with options to support CUDA or ROCm and package the warmed cache into a tarball for distribution.
* **Human-Readable Output:** Presents kernel information in well-formatted tables with human-readable file sizes and modification times.

## Installation

### Prerequisites

* Python 3.9 or higher.
* Triton
* [Podman](https://podman.io/) (required for the `tcm warm` command).

### Installation Steps

1.  **Install in editable mode:**

    ```bash
    pip install -e .
    ```

    This command installs TCM and its dependencies (like `typer`, `rich`, `sqlalchemy`, `pydantic`) and makes the `tcm` command-line tool available in your environment.

---

## Quick Start

1.  **Index your Triton kernel cache:**

    ```bash
    tcm index --cache-dir ~/.triton/cache # Or your custom cache path
    ```

2.  **List kernels by backend:**

    ```bash
    tcm list --backend cuda
    ```

---

## Usage

The `tcm` command-line interface provides several subcommands.

### Indexing the Cache

The `index` command scans the specified Triton cache directory (or the default `~/.triton/cache`) and populates the local database with kernel metadata.

```bash
tcm index [OPTIONS]
````

**Options:**

  * `--cache-dir PATH`: Specify the Triton cache directory. If not provided, it defaults to `~/.triton/cache` (or `~/.local/share/triton-cache-manager` on Linux).

**Example:**

```bash
tcm index --cache-dir /path/to/my/triton/cache
```

### Listing Kernels (Search)

The `list` command allows you to search and display indexed kernels based on various criteria.

```bash
tcm list [OPTIONS]
```

**Options:**

  * `--name -n TEXT`: Filter by kernel name (exact match).
  * `--backend -b TEXT`: Filter by backend (e.g., `'cuda'`, `'rocm'`).
  * `--arch -a TEXT`: Filter by architecture (e.g., `'80'` for CUDA, `'gfx90a'` for ROCm).
  * `--older-than TEXT`: Show kernels older than a specified duration (e.g., `'7d'` for 7 days, `'2w'` for 2 weeks).
  * `--younger-than TEXT`: Show kernels younger than a specified duration (e.g., `'14d'` for 14 days, `'1w'` for 1 week).
  * `--cache-dir PATH`: Specify the Triton cache directory to search within (uses default if not provided).

**Examples:**

  * List all CUDA kernels:
    ```bash
    tcm list --backend cuda
    ```
  * Find kernels named `my_custom_kernel` that are older than 30 days:
    ```bash
    tcm list --name my_custom_kernel --older-than 30d
    ```
  * List ROCm kernels for a specific architecture:
    ```bash
    tcm list --backend rocm --arch gfx90a
    ```

### Pruning Kernels

The `prune` command removes kernel files from the cache based on specified filters. It can perform partial (IR-only) or full deletions, and also deduplicate kernels.

```bash
tcm prune [OPTIONS]
```

**Options:**

  * `--name -n TEXT`: Filter by kernel name.
  * `--backend -b TEXT`: Filter by backend.
  * `--arch -a TEXT`: Filter by architecture.
  * `--older-than TEXT`: Prune kernels older than a specified duration.
  * `--younger-than TEXT`: Prune kernels younger than a specified duration.
  * `--full`: Remove the entire kernel directory (default is to only remove IR files).
  * `--deduplicate`: Delete older duplicate kernels, keeping only the newest. *Note: Other filter options and `--full` are ignored when `--deduplicate` is used.*
  * `-y, --yes`: Skip the confirmation prompt.
  * `--cache-dir PATH`: Specify the Triton cache directory (uses default if not provided).

**Examples:**

  * Prune IR files of all kernels older than 90 days:
    ```bash
    tcm prune --older-than 90d
    ```
  * Fully remove all "unstable\_kernel" entries:
    ```bash
    tcm prune --name unstable_kernel --full -y
    ```
  * Deduplicate all kernels, removing older duplicates:
    ```bash
    tcm prune --deduplicate -y
    ```

### Warming the Cache

The `warm` command utilizes Podman to run a container that warms up the vLLM cache for a specified model. It can then optionally package the warmed cache into a tarball.

```bash
tcm warm [OPTIONS]
```

**Options:**

  * `--model -m TEXT`: The Hugging Face model to use for warming the cache (default: `facebook/opt-125m`).
  * `--output -o PATH`: The path to save the packaged cache archive (default: `warmed_cache.tar.gz`).
  * `--host-cache-dir TEXT`: The directory on the host machine where the vLLM cache will be stored (default: `./`).
  * `--hugging-face-token TEXT`: Your Hugging Face token for accessing models.
  * `--vllm_cache_dir TEXT`: The vLLM cache directory *inside the container* (default: `/root/.cache/vllm/`).
  * `--tarball`: Create a gzipped tarball of the warmed vLLM cache.
  * `--rocm`: Warm the vLLM cache for ROCm GPUs (default is CUDA).

**Examples:**

  * Warm cache for a model and create a tarball:
    ```bash
    tcm warm --model EleutherAI/gpt-neo-125M --tarball --output my_gpt_neo_cache.tar.gz
    ```
  * Warm cache for a ROCm system:
    ```bash
    tcm warm --model Llama-3-8B --rocm --hugging-face-token hf_YOUR_TOKEN
    ```

-----

## Requirements

The core dependencies for TCM are listed in `requirements.txt`:

  * `typer`
  * `rich`
  * `pydantic`
  * `pydantic-settings`
  * `structlog`
  * `sqlalchemy`

These are automatically installed when you follow the installation steps.

-----

## Project Structure

The project is organized into logical modules:

  * `tcm/triton_cache_manager/cli`: Contains the main command-line interface logic.
  * `tcm/triton_cache_manager/services`: Implements the core business logic for indexing, searching, pruning, and warming.
  * `tcm/triton_cache_manager/data`: Handles data access, including database interactions (SQLite with SQLAlchemy) and cache repository operations.
  * `tcm/triton_cache_manager/models`: Defines data transfer objects (DTOs) and Pydantic models for kernel metadata and search criteria.
  * `tcm/triton_cache_manager/plugins`: Provides an extensible architecture for supporting different Triton backends (e.g., CUDA, ROCm) and their specific file types.
  * `tcm/triton_cache_manager/utils`: Contains utility functions for logging, path management, and data formatting.

