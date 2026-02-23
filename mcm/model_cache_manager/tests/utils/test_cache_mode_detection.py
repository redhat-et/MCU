"""
Unit tests for cache mode detection logic.
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from model_cache_manager.utils.utils import detect_cache_mode
from model_cache_manager.utils.mcm_constants import MODE_TRITON, MODE_VLLM, MODE_VLLM_LEGACY


class TestCacheModeDetection(unittest.TestCase):
    """Test suite for cache mode detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_detect_non_existent_directory(self):
        """Test detection on non-existent directory defaults to triton."""
        non_existent = self.temp_dir / "does_not_exist"
        mode = detect_cache_mode(non_existent)
        self.assertEqual(mode, MODE_TRITON)

    def test_detect_empty_directory(self):
        """Test detection on empty directory defaults to triton."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir(parents=True)
        mode = detect_cache_mode(empty_dir)
        self.assertEqual(mode, MODE_TRITON)

    def test_detect_triton_cache_structure(self):
        """Test detection of standard Triton cache structure."""
        # Create a typical Triton cache structure
        triton_dir = self.temp_dir / "triton_cache"
        kernel_dir = triton_dir / "some_kernel_hash"
        kernel_dir.mkdir(parents=True)

        # Add a typical triton kernel metadata file
        (kernel_dir / "metadata.json").touch()

        mode = detect_cache_mode(triton_dir)
        self.assertEqual(mode, MODE_TRITON)

    def test_detect_vllm_legacy_cache_structure(self):
        """Test detection of legacy vLLM cache structure."""
        # Create legacy vLLM structure: torch_compile_cache/<hash>/rank_x_y/triton_cache/
        vllm_dir = self.temp_dir / "vllm_legacy"
        torch_compile = vllm_dir / "torch_compile_cache"
        hash_dir = torch_compile / "some_vllm_hash"
        rank_dir = hash_dir / "rank_0_0"
        triton_cache = rank_dir / "triton_cache"
        triton_cache.mkdir(parents=True)

        # Add a kernel directory in triton_cache
        kernel_dir = triton_cache / "kernel_hash"
        kernel_dir.mkdir(parents=True)

        mode = detect_cache_mode(vllm_dir)
        self.assertEqual(mode, MODE_VLLM_LEGACY)

    def test_detect_vllm_new_cache_structure(self):
        """Test detection of new vLLM cache structure with artifact_compile_range."""
        # Create vLLM structure: torch_compile_cache/<hash>/rank_x_y/...
        vllm_dir = self.temp_dir / "vllm_new"
        torch_compile = vllm_dir / "torch_compile_cache"
        hash_dir = torch_compile / "some_vllm_hash"
        rank_dir = hash_dir / "rank_0_0"
        backbone_dir = rank_dir / "backbone"
        artifact_dir = backbone_dir / "artifact_compile_range_0"
        triton_dir = artifact_dir / "triton"
        triton_dir.mkdir(parents=True)

        # Add a subdirectory in triton (e.g., "0")
        sub_dir = triton_dir / "0"
        sub_dir.mkdir(parents=True)

        # Add kernel files
        kernel_dir = sub_dir / "kernel_hash"
        kernel_dir.mkdir(parents=True)

        mode = detect_cache_mode(vllm_dir)
        self.assertEqual(mode, MODE_VLLM)

    def test_detect_mixed_structure_prefers_new_vllm(self):
        """Test that new vLLM structure is preferred if both old and new exist."""
        # Create a directory with both structures
        mixed_dir = self.temp_dir / "mixed"
        torch_compile = mixed_dir / "torch_compile_cache"
        hash_dir = torch_compile / "some_hash"
        rank_dir = hash_dir / "rank_0_0"

        # Add legacy structure
        triton_cache = rank_dir / "triton_cache"
        triton_cache.mkdir(parents=True)

        # Add new structure
        backbone_dir = rank_dir / "backbone"
        artifact_dir = backbone_dir / "artifact_compile_range_0"
        triton_dir = artifact_dir / "triton"
        triton_dir.mkdir(parents=True)

        # Should detect as new vLLM since we check for new structure first
        mode = detect_cache_mode(mixed_dir)
        self.assertEqual(mode, MODE_VLLM)

    def test_detect_vllm_new_with_best_config(self):
        """Test detection of new vLLM structure with best config files."""
        # Create new vLLM structure with best config
        vllm_dir = self.temp_dir / "vllm_with_config"
        torch_compile = vllm_dir / "torch_compile_cache"
        hash_dir = torch_compile / "4405297553"
        rank_dir = hash_dir / "rank_0_0"
        backbone_dir = rank_dir / "backbone"
        artifact_dir = backbone_dir / "artifact_compile_range_None_subgraph_0"
        triton_dir = artifact_dir / "triton"
        triton_dir.mkdir(parents=True)

        # Add best config file
        config_dir = artifact_dir / "tc"
        config_dir.mkdir(parents=True)
        cfg_file = config_dir / "c9a602056cc9b1.best_config"
        cfg_file.write_text('{"XBLOCK": 512, "num_warps": 8}')

        mode = detect_cache_mode(vllm_dir)
        self.assertEqual(mode, MODE_VLLM)

    def test_detect_multiple_rank_directories(self):
        """Test detection with multiple rank directories."""
        # Test legacy vLLM with multiple ranks
        vllm_dir = self.temp_dir / "vllm_multi_rank"
        torch_compile = vllm_dir / "torch_compile_cache"
        hash_dir = torch_compile / "hash123"

        # Create multiple rank directories
        for rank in ["rank_0_0", "rank_0_1", "rank_1_0"]:
            rank_dir = hash_dir / rank
            triton_cache = rank_dir / "triton_cache"
            triton_cache.mkdir(parents=True)

        mode = detect_cache_mode(vllm_dir)
        self.assertEqual(mode, MODE_VLLM_LEGACY)

    def test_detect_incomplete_vllm_structure(self):
        """Test that incomplete vLLM structure defaults to triton."""
        # Create structure missing critical components
        incomplete_dir = self.temp_dir / "incomplete"
        torch_compile = incomplete_dir / "torch_compile_cache"
        hash_dir = torch_compile / "some_hash"
        rank_dir = hash_dir / "rank_0_0"
        rank_dir.mkdir(parents=True)
        # No triton_cache or backbone directory

        mode = detect_cache_mode(incomplete_dir)
        self.assertEqual(mode, MODE_TRITON)

    def test_detect_vllm_structure_multiple_hashes(self):
        """Test detection with vLLM structure having multiple hash directories."""
        cache_dir = self.temp_dir / "vllm_multi_hash"
        torch_compile_dir = cache_dir / "torch_compile_cache"

        # Create multiple hash directories
        for hash_name in ["hash1", "hash2"]:
            hash_dir = torch_compile_dir / hash_name
            rank_dir = hash_dir / "rank_0_0"
            triton_cache = rank_dir / "triton_cache"
            triton_cache.mkdir(parents=True)

        mode = detect_cache_mode(cache_dir)
        self.assertEqual(mode, MODE_VLLM_LEGACY)

    def test_detect_triton_structure_with_json_files(self):
        """Test detection with triton cache structure containing JSON files."""
        cache_dir = self.temp_dir / "triton_cache"
        kernel_dir = cache_dir / "triton_kernel_abc123"
        kernel_dir.mkdir(parents=True)

        # Create a JSON file that looks like triton metadata
        json_file = kernel_dir / "kernel_metadata.json"
        json_file.write_text('{"name": "test_kernel"}')

        mode = detect_cache_mode(cache_dir)
        self.assertEqual(mode, MODE_TRITON)

    def test_detect_empty_torch_compile_cache(self):
        """Test detection with torch_compile_cache directory but no contents."""
        cache_dir = self.temp_dir / "vllm_empty"
        torch_compile_dir = cache_dir / "torch_compile_cache"
        torch_compile_dir.mkdir(parents=True)

        mode = detect_cache_mode(cache_dir)
        self.assertEqual(mode, MODE_TRITON)

    def test_detect_rank_dirs_not_starting_with_rank(self):
        """Test detection with directories that don't start with 'rank'."""
        cache_dir = self.temp_dir / "vllm_bad_rank"
        torch_compile_dir = cache_dir / "torch_compile_cache"
        hash_dir = torch_compile_dir / "abc123def456"
        not_rank_dir = hash_dir / "not_rank_dir"
        triton_cache = not_rank_dir / "triton_cache"
        triton_cache.mkdir(parents=True)

        mode = detect_cache_mode(cache_dir)
        self.assertEqual(mode, MODE_TRITON)

    def test_detect_case_sensitivity(self):
        """Test detection is case sensitive for directory names."""
        cache_dir = self.temp_dir / "vllm_case"
        torch_compile_dir = cache_dir / "Torch_compile_cache"  # Wrong case
        hash_dir = torch_compile_dir / "abc123def456"
        rank_dir = hash_dir / "Rank_0_0"  # Wrong case
        triton_cache = rank_dir / "triton_cache"
        triton_cache.mkdir(parents=True)

        mode = detect_cache_mode(cache_dir)
        self.assertEqual(mode, MODE_TRITON)

    def test_detect_files_not_directories(self):
        """Test detection when torch_compile_cache is a file, not directory."""
        cache_dir = self.temp_dir / "vllm_file"
        cache_dir.mkdir()

        # Create torch_compile_cache as a file instead of directory
        torch_compile_file = cache_dir / "torch_compile_cache"
        torch_compile_file.write_text("not a directory")

        mode = detect_cache_mode(cache_dir)
        self.assertEqual(mode, MODE_TRITON)

    def test_detect_vllm_structure_no_rank_dirs(self):
        """Test detection with torch_compile_cache but no rank directories."""
        cache_dir = self.temp_dir / "vllm_no_rank"
        torch_compile_dir = cache_dir / "torch_compile_cache"
        hash_dir = torch_compile_dir / "abc123def456"
        # Create hash directory but no rank dirs
        hash_dir.mkdir(parents=True)

        mode = detect_cache_mode(cache_dir)
        self.assertEqual(mode, MODE_TRITON)

    def test_detect_vllm_binary_cache_structure(self):
        """Test detection of new vLLM cache structure with binary artifacts."""
        # Binary artifacts are files (not directories) named artifact_compile_range_*
        vllm_dir = self.temp_dir / "vllm_binary"
        torch_compile = vllm_dir / "torch_compile_cache"
        hash_dir = torch_compile / "some_vllm_hash"
        rank_dir = hash_dir / "rank_0_0"
        backbone_dir = rank_dir / "backbone"
        backbone_dir.mkdir(parents=True)

        # Create a binary artifact file (not a directory)
        binary_artifact = backbone_dir / "artifact_compile_range_0"
        binary_artifact.write_bytes(b"\x00binary data")

        mode = detect_cache_mode(vllm_dir)
        self.assertEqual(mode, MODE_VLLM)

    def test_detect_mixed_legacy_and_triton_vllm_takes_precedence(self):
        """Test detection when both vllm-legacy and triton structures exist."""
        cache_dir = self.temp_dir / "mixed_cache"

        # Create vLLM legacy structure
        torch_compile_dir = cache_dir / "torch_compile_cache"
        hash_dir = torch_compile_dir / "abc123def456"
        rank_dir = hash_dir / "rank_0_0"
        triton_cache = rank_dir / "triton_cache"
        triton_cache.mkdir(parents=True)

        # Also create triton-like structure at root level
        triton_kernel_dir = cache_dir / "triton_kernel_def789"
        triton_kernel_dir.mkdir(parents=True)
        json_file = triton_kernel_dir / "metadata.json"
        json_file.write_text('{"name": "triton_kernel"}')

        mode = detect_cache_mode(cache_dir)
        self.assertEqual(mode, MODE_VLLM_LEGACY)


if __name__ == "__main__":
    unittest.main()
