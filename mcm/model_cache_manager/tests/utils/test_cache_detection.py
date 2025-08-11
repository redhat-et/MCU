"""
Unit tests for cache mode auto-detection functionality.
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from model_cache_manager.utils.utils import detect_cache_mode


class TestCacheDetection(unittest.TestCase):
    """Test suite for cache mode auto-detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_detect_cache_mode_nonexistent_directory(self):
        """Test detection with non-existent directory returns triton."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        result = detect_cache_mode(nonexistent_dir)
        self.assertEqual(result, "triton")

    def test_detect_cache_mode_empty_directory(self):
        """Test detection with empty directory returns triton."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()

        result = detect_cache_mode(empty_dir)
        self.assertEqual(result, "triton")

    def test_detect_cache_mode_vllm_structure(self):
        """Test detection with vLLM cache structure returns vllm."""
        # Create vLLM structure: cache_dir/torch_compile_cache/hash/rank0_0/triton_cache
        cache_dir = self.temp_dir / "vllm_cache"
        torch_compile_dir = cache_dir / "torch_compile_cache"
        hash_dir = torch_compile_dir / "abc123def456"
        rank_dir = hash_dir / "rank0_0"
        triton_cache = rank_dir / "triton_cache"

        triton_cache.mkdir(parents=True)

        result = detect_cache_mode(cache_dir)
        self.assertEqual(result, "vllm")

    def test_detect_cache_mode_vllm_structure_multiple_hashes(self):
        """Test detection with vLLM structure having multiple hash directories."""
        cache_dir = self.temp_dir / "vllm_cache"
        torch_compile_dir = cache_dir / "torch_compile_cache"

        # Create multiple hash directories
        hash_dir1 = torch_compile_dir / "hash1"
        hash_dir2 = torch_compile_dir / "hash2"

        rank_dir1 = hash_dir1 / "rank0_0"
        rank_dir2 = hash_dir2 / "rank1_0"

        triton_cache1 = rank_dir1 / "triton_cache"
        triton_cache2 = rank_dir2 / "triton_cache"

        triton_cache1.mkdir(parents=True)
        triton_cache2.mkdir(parents=True)

        result = detect_cache_mode(cache_dir)
        self.assertEqual(result, "vllm")

    def test_detect_cache_mode_vllm_structure_without_triton_cache(self):
        """Test detection with vLLM structure but no triton_cache subdirectory."""
        cache_dir = self.temp_dir / "vllm_cache"
        torch_compile_dir = cache_dir / "torch_compile_cache"
        hash_dir = torch_compile_dir / "abc123def456"
        rank_dir = hash_dir / "rank0_0"

        # Create rank directory but no triton_cache
        rank_dir.mkdir(parents=True)

        result = detect_cache_mode(cache_dir)
        self.assertEqual(result, "triton")

    def test_detect_cache_mode_vllm_structure_no_rank_dirs(self):
        """Test detection with torch_compile_cache but no rank directories."""
        cache_dir = self.temp_dir / "vllm_cache"
        torch_compile_dir = cache_dir / "torch_compile_cache"
        hash_dir = torch_compile_dir / "abc123def456"

        # Create hash directory but no rank dirs
        hash_dir.mkdir(parents=True)

        result = detect_cache_mode(cache_dir)
        self.assertEqual(result, "triton")

    def test_detect_cache_mode_triton_structure_with_json_files(self):
        """Test detection with triton cache structure containing JSON files."""
        cache_dir = self.temp_dir / "triton_cache"
        kernel_dir = cache_dir / "triton_kernel_abc123"

        kernel_dir.mkdir(parents=True)

        # Create a JSON file that looks like triton metadata
        json_file = kernel_dir / "kernel_metadata.json"
        json_file.write_text('{"name": "test_kernel"}')

        result = detect_cache_mode(cache_dir)
        self.assertEqual(result, "triton")

    def test_detect_cache_mode_mixed_structure_vllm_takes_precedence(self):
        """Test detection when both structures exist - vLLM should take precedence."""
        cache_dir = self.temp_dir / "mixed_cache"

        # Create vLLM structure
        torch_compile_dir = cache_dir / "torch_compile_cache"
        hash_dir = torch_compile_dir / "abc123def456"
        rank_dir = hash_dir / "rank0_0"
        triton_cache = rank_dir / "triton_cache"
        triton_cache.mkdir(parents=True)

        # Also create triton-like structure
        triton_kernel_dir = cache_dir / "triton_kernel_def789"
        triton_kernel_dir.mkdir(parents=True)
        json_file = triton_kernel_dir / "metadata.json"
        json_file.write_text('{"name": "triton_kernel"}')

        result = detect_cache_mode(cache_dir)
        self.assertEqual(result, "vllm")

    def test_detect_cache_mode_torch_compile_cache_empty(self):
        """Test detection with torch_compile_cache directory but no contents."""
        cache_dir = self.temp_dir / "vllm_cache"
        torch_compile_dir = cache_dir / "torch_compile_cache"
        torch_compile_dir.mkdir(parents=True)

        result = detect_cache_mode(cache_dir)
        self.assertEqual(result, "triton")

    def test_detect_cache_mode_rank_dirs_not_starting_with_rank(self):
        """Test detection with directories that don't start with 'rank'."""
        cache_dir = self.temp_dir / "vllm_cache"
        torch_compile_dir = cache_dir / "torch_compile_cache"
        hash_dir = torch_compile_dir / "abc123def456"
        not_rank_dir = hash_dir / "not_rank_dir"
        triton_cache = not_rank_dir / "triton_cache"

        triton_cache.mkdir(parents=True)

        result = detect_cache_mode(cache_dir)
        self.assertEqual(result, "triton")

    def test_detect_cache_mode_complex_vllm_structure(self):
        """Test detection with complex vLLM structure having multiple ranks."""
        cache_dir = self.temp_dir / "vllm_cache"
        torch_compile_dir = cache_dir / "torch_compile_cache"
        hash_dir = torch_compile_dir / "abc123def456"

        # Create multiple rank directories
        rank_dirs = ["rank0_0", "rank0_1", "rank1_0", "rank1_1"]
        for rank_name in rank_dirs:
            rank_dir = hash_dir / rank_name
            triton_cache = rank_dir / "triton_cache"
            triton_cache.mkdir(parents=True)

        result = detect_cache_mode(cache_dir)
        self.assertEqual(result, "vllm")

    def test_detect_cache_mode_case_sensitivity(self):
        """Test detection is case sensitive for directory names."""
        cache_dir = self.temp_dir / "vllm_cache"
        torch_compile_dir = cache_dir / "Torch_compile_cache"  # Wrong case
        hash_dir = torch_compile_dir / "abc123def456"
        rank_dir = hash_dir / "Rank0_0"  # Wrong case
        triton_cache = rank_dir / "triton_cache"

        triton_cache.mkdir(parents=True)

        result = detect_cache_mode(cache_dir)
        self.assertEqual(result, "triton")

    def test_detect_cache_mode_files_not_directories(self):
        """Test detection when torch_compile_cache is a file, not directory."""
        cache_dir = self.temp_dir / "vllm_cache"
        cache_dir.mkdir()

        # Create torch_compile_cache as a file instead of directory
        torch_compile_file = cache_dir / "torch_compile_cache"
        torch_compile_file.write_text("not a directory")

        result = detect_cache_mode(cache_dir)
        self.assertEqual(result, "triton")


if __name__ == "__main__":
    unittest.main()
