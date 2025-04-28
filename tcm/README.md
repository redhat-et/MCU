# Triton Cache Manager (TCM)

A lightweight CLI for **indexing, searching, and more for Triton GPU-kernel caches**.

---

## Installation

```bash
pip install -e .
```

---

## Quick start

```bash
# Index triton kernel cache
tcm index --cache-dir ~/.triton

# search kernels by backend
tcm list --backend cuda

```

---

## Requirements

- Python ≥ 3.9
- Triton
