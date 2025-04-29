"""
Unit tests for the kernel_validator module.

Tests the deserialization of kernel metadata from JSON with focus on validation.
"""

import unittest
from pathlib import Path
from typing import Dict, cast
from unittest.mock import Mock, patch

from triton_cache_manager.data.kernel_validator import deserialize_kernel
from triton_cache_manager.plugins.base import KernelBackendPlugin


class CudaPluginMock(Mock):
    """Mock implementation of KernelBackendPlugin for testing."""

    backend = "cuda"

    def relevant_extensions(self) -> Dict[str, str]:
        """Return mock file extensions map."""
        return {".ptx": "ptx", ".cubin": "cubin"}


class TestKernelValidator(unittest.TestCase):
    """Tests for kernel metadata validation functionality."""

    def setUp(self):
        """Set up common test fixtures."""
        self.test_dir = Path("/test/dir")

        cuda_plugin = CudaPluginMock()
        self.plugins = {"cuda": cast(KernelBackendPlugin, cuda_plugin)}

        self.mock_files = [
            Mock(suffix=".json", path=Path("/test/dir/metadata.json")),
            Mock(suffix=".ptx", path=Path("/test/dir/kernel.ptx")),
        ]
        for mock_file in self.mock_files:
            mock_file.stat = Mock(return_value=Mock(st_size=1024))

    def test_valid_kernel_json(self):
        """Test deserializing a valid kernel JSON."""
        valid_data = {
            "name": "test_kernel",
            "target": {"backend": "cuda", "arch": 80, "warp_size": 32},
            "num_warps": 4,
            "num_stages": 2,
            "triton_version": "3.3.0",
            "tmem_size": 0,
            "maxnreg": 1,
            "cluster_dims": [1, 1, 1],
            "debug": False,
            "shared": 1234
        }
        with patch("pathlib.Path.iterdir", return_value=self.mock_files):
            kernel = deserialize_kernel(
                valid_data, "1234abcd", self.test_dir, self.plugins
            )

        self.assertIsNotNone(kernel)
        self.assertEqual(kernel.hash, "1234abcd")
        self.assertEqual(kernel.name, "test_kernel")
        self.assertEqual(kernel.backend, "cuda")
        self.assertEqual(kernel.arch, 80)
        self.assertEqual(kernel.warp_size, 32)

        self.assertEqual(len(kernel.files), 2)
        self.assertEqual(kernel.files[0].file_type, "metadata")
        self.assertEqual(kernel.files[1].file_type, "ptx")

    def test_corrupt_json(self):
        """Test rejection of corrupt/invalid JSON."""
        invalid_data = {
            "name": 123,  # Should be string
            "target": "not_a_dict",  # Should be a dictionary
            "num_warps": "cinque",  # Should be an integer
        }

        with patch("pathlib.Path.iterdir", return_value=self.mock_files):
            kernel = deserialize_kernel(
                invalid_data, "1234abcd", self.test_dir, self.plugins
            )

        self.assertIsNone(kernel)

    def test_unrelated_json(self):
        """Test rejection of JSON that's not kernel-related."""
        unrelated_data = {
            "type": "user_config",
            "username": "sangiorgi",
            "preferences": {"theme": "italian", "notifications": True},
        }

        with patch("pathlib.Path.iterdir", return_value=self.mock_files):
            kernel = deserialize_kernel(
                unrelated_data, "1234abcd", self.test_dir, self.plugins
            )

        self.assertIsNone(kernel)


if __name__ == "__main__":
    unittest.main()
