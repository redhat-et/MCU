"""
Triton-Util 🔱 – Utility functions for writing Triton GPU kernels with less friction.

This package provides high-level abstractions and helpers for writing
fast and readable Triton code, reducing the need for repetitive and error-prone
index calculations.

Features:
- **Coding utilities**: Chunking, masking, offset computation, and bulk load/store helpers.
- **Debugging utilities**: Convenient debugging hooks like `print_once` and `breakpoint_if`
  that make inspecting kernel behavior simpler and less intrusive.

The utilities are designed to be:
- Minimal and interoperable: fully compatible with native Triton code.
- Expressive: match how you actually think about GPU data access patterns.
- Progressive: use as little or as much of the library as needed.

Example usage:
    >>> load_2d(ptr, sz0, sz1, n0, n1, max0, max1, stride0)

instead of:
    >>> offs0 = n0 * sz0 + tl.arange(0, sz0)
    >>> offs1 = n1 * sz1 + tl.arange(0, sz1)
    >>> offs = offs0[:, None] * stride0 + offs1[None, :] * stride1
    >>> mask = (offs0[:, None] < max0) & (offs1[None, :] < max1)
    >>> tl.load(ptr + offs, mask)

For documentation, examples, and community support, see:
- GitHub: https://github.com/cuda-mode/triton-util
- Discord: https://discord.gg/cudamode (Triton channel)

Author: Umer Hadil
"""

from .debugging import *
from .coding import *
