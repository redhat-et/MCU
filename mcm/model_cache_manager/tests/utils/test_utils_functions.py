"""
Unit tests for utility functions in utils.py module.
"""

import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile
import shutil

from model_cache_manager.utils.utils import (
    get_temp_extraction_dir,
    _find_kernel_dirs_in_aot_cache,
    find_vllm_kernel_dirs,
)


class TestGetTempExtractionDir(unittest.TestCase):
    """Tests for get_temp_extraction_dir function."""

    def test_creates_directory_if_not_exists(self):
        """Test that the function creates the directory if it doesn't exist."""
        with patch("getpass.getuser", return_value="testuser"):
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                result = get_temp_extraction_dir()

                # Verify mkdir was called with correct arguments
                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
                self.assertEqual(result, Path("/tmp/mcm_testuser"))

    def test_returns_existing_directory(self):
        """Test that the function returns existing directory without error."""
        # Create actual temp directory for this test
        temp_test_dir = Path(tempfile.mkdtemp())
        try:
            with patch("getpass.getuser", return_value="testuser"):
                with patch("pathlib.Path.__truediv__", return_value=temp_test_dir):
                    result = get_temp_extraction_dir()

                    # Should return the path even if it already exists
                    self.assertTrue(result.exists())
        finally:
            shutil.rmtree(temp_test_dir, ignore_errors=True)

    def test_uses_current_username(self):
        """Test that the function uses the current user's username."""
        with patch("getpass.getuser", return_value="john_doe"):
            result = get_temp_extraction_dir()

            self.assertEqual(result, Path("/tmp/mcm_john_doe"))
            self.assertIn("john_doe", str(result))

    def test_path_format(self):
        """Test that the returned path has the correct format."""
        with patch("getpass.getuser", return_value="alice"):
            result = get_temp_extraction_dir()

            self.assertEqual(result.parent, Path("/tmp"))
            self.assertEqual(result.name, "mcm_alice")
            self.assertTrue(str(result).startswith("/tmp/mcm_"))

    def test_different_users_get_different_dirs(self):
        """Test that different users get different directories."""
        with patch("getpass.getuser", return_value="user1"):
            result1 = get_temp_extraction_dir()

        with patch("getpass.getuser", return_value="user2"):
            result2 = get_temp_extraction_dir()

        self.assertNotEqual(result1, result2)
        self.assertEqual(result1, Path("/tmp/mcm_user1"))
        self.assertEqual(result2, Path("/tmp/mcm_user2"))

    def test_idempotent(self):
        """Test that calling the function multiple times returns the same path."""
        with patch("getpass.getuser", return_value="testuser"):
            result1 = get_temp_extraction_dir()
            result2 = get_temp_extraction_dir()

            self.assertEqual(result1, result2)


class TestFindKernelDirsInAotCache(unittest.TestCase):
    """Tests for _find_kernel_dirs_in_aot_cache function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_aot_kernel(self, vllm_hash, device, kernel_hash):
        """Create a kernel directory in the AOT layout and return its path."""
        kernel_dir = (
            self.temp_dir
            / "torch_compile_cache"
            / "torch_aot_compile"
            / vllm_hash
            / "inductor_cache"
            / "triton"
            / device
            / kernel_hash
        )
        kernel_dir.mkdir(parents=True)
        return kernel_dir

    def test_finds_kernel_in_aot_cache(self):
        """Test finding a kernel directory in the AOT layout."""
        self._create_aot_kernel("hash1", "0", "KERNEL_ABC")

        result = _find_kernel_dirs_in_aot_cache(
            self.temp_dir, "hash1", "KERNEL_ABC"
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "KERNEL_ABC")

    def test_returns_empty_when_hash_missing(self):
        """Test returns empty list when vllm_hash doesn't exist."""
        self._create_aot_kernel("hash1", "0", "KERNEL_ABC")

        result = _find_kernel_dirs_in_aot_cache(
            self.temp_dir, "nonexistent", "KERNEL_ABC"
        )
        self.assertEqual(result, [])

    def test_returns_empty_when_kernel_missing(self):
        """Test returns empty list when kernel hash doesn't exist."""
        self._create_aot_kernel("hash1", "0", "KERNEL_ABC")

        result = _find_kernel_dirs_in_aot_cache(
            self.temp_dir, "hash1", "NONEXISTENT"
        )
        self.assertEqual(result, [])

    def test_returns_empty_when_no_triton_dir(self):
        """Test returns empty list when inductor_cache/triton doesn't exist."""
        (
            self.temp_dir
            / "torch_compile_cache"
            / "torch_aot_compile"
            / "hash1"
            / "inductor_cache"
            / "fxgraph"
        ).mkdir(parents=True)

        result = _find_kernel_dirs_in_aot_cache(
            self.temp_dir, "hash1", "KERNEL_ABC"
        )
        self.assertEqual(result, [])

    def test_finds_kernel_across_devices(self):
        """Test finding a kernel that appears under multiple device subdirs."""
        self._create_aot_kernel("hash1", "0", "KERNEL_ABC")
        self._create_aot_kernel("hash1", "1", "KERNEL_ABC")

        result = _find_kernel_dirs_in_aot_cache(
            self.temp_dir, "hash1", "KERNEL_ABC"
        )
        self.assertEqual(len(result), 2)


class TestFindVllmKernelDirsAot(unittest.TestCase):
    """Tests for find_vllm_kernel_dirs with AOT layout."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_finds_aot_kernel_when_no_compile_cache(self):
        """Test finds AOT kernels when torch_compile_cache has no matching hash."""
        kernel_dir = (
            self.temp_dir
            / "torch_compile_cache"
            / "torch_aot_compile"
            / "hash1"
            / "inductor_cache"
            / "triton"
            / "0"
            / "KERNEL_XYZ"
        )
        kernel_dir.mkdir(parents=True)

        result = find_vllm_kernel_dirs(
            self.temp_dir, "hash1", "KERNEL_XYZ"
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "KERNEL_XYZ")

    def test_combines_both_layouts(self):
        """Test finds kernels from both torch_compile_cache and AOT."""
        # AOT layout
        aot_kernel = (
            self.temp_dir
            / "torch_compile_cache"
            / "torch_aot_compile"
            / "hash1"
            / "inductor_cache"
            / "triton"
            / "0"
            / "KERNEL_A"
        )
        aot_kernel.mkdir(parents=True)

        # torch_compile_cache layout
        tcc_kernel = (
            self.temp_dir
            / "torch_compile_cache"
            / "hash1"
            / "rank_0_0"
            / "backbone"
            / "artifact_compile_range_0"
            / "triton"
            / "0"
            / "KERNEL_B"
        )
        tcc_kernel.mkdir(parents=True)

        result_a = find_vllm_kernel_dirs(
            self.temp_dir, "hash1", "KERNEL_A"
        )
        result_b = find_vllm_kernel_dirs(
            self.temp_dir, "hash1", "KERNEL_B"
        )
        self.assertEqual(len(result_a), 1)
        self.assertEqual(len(result_b), 1)

    def test_returns_empty_for_nonexistent_hash(self):
        """Test returns empty when neither layout has the hash."""
        result = find_vllm_kernel_dirs(
            self.temp_dir, "nonexistent", "KERNEL_A"
        )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
