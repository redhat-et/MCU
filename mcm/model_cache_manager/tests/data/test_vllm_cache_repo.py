"""
Unit tests for the VllmCacheRepository.
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
import tempfile
import os

from model_cache_manager.data.cache_repo import VllmCacheRepository, CacheRepository
from model_cache_manager.models.kernel import Kernel


class TestVllmCacheRepository(unittest.TestCase):
    """Test suite for the VllmCacheRepository."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.vllm_cache_root = self.temp_dir / "vllm_cache"
        self.vllm_cache_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_with_existing_directory(self):
        """Test initializing VllmCacheRepository with existing directory."""
        repo = VllmCacheRepository(self.vllm_cache_root)
        self.assertEqual(repo.root, self.vllm_cache_root)

    def test_init_with_nonexistent_directory(self):
        """Test initializing VllmCacheRepository with non-existent directory."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        with self.assertRaises(FileNotFoundError):
            VllmCacheRepository(nonexistent_dir)

    def test_init_with_default_directory(self):
        """Test initializing VllmCacheRepository with default directory."""
        with patch('pathlib.Path.home') as mock_home:
            mock_home.return_value = self.temp_dir
            default_vllm_cache = self.temp_dir / ".cache" / "vllm"
            default_vllm_cache.mkdir(parents=True, exist_ok=True)

            repo = VllmCacheRepository()
            self.assertEqual(repo.root, default_vllm_cache)

    def test_find_torch_compile_cache_dirs_empty(self):
        """Test finding torch compile cache directories when none exist."""
        repo = VllmCacheRepository(self.vllm_cache_root)
        dirs = list(repo._find_torch_compile_cache_dirs())
        self.assertEqual(len(dirs), 0)

    def test_find_torch_compile_cache_dirs_with_dirs(self):
        """Test finding torch compile cache directories when they exist."""
        # Create mock vLLM structure
        torch_compile_dir = self.vllm_cache_root / "torch_compile_cache"
        torch_compile_dir.mkdir()

        hash_dir1 = torch_compile_dir / "hash123abc"
        hash_dir2 = torch_compile_dir / "hash456def"
        hash_dir1.mkdir()
        hash_dir2.mkdir()

        repo = VllmCacheRepository(self.vllm_cache_root)
        dirs = list(repo._find_torch_compile_cache_dirs())

        self.assertEqual(len(dirs), 2)
        hash_names = [hash_name for hash_name, _ in dirs]
        self.assertIn("hash123abc", hash_names)
        self.assertIn("hash456def", hash_names)

    def test_find_rank_dirs_empty(self):
        """Test finding rank directories when none exist."""
        hash_dir = self.vllm_cache_root / "test_hash"
        hash_dir.mkdir()

        repo = VllmCacheRepository(self.vllm_cache_root)
        rank_dirs = list(repo._find_rank_dirs(hash_dir))
        self.assertEqual(len(rank_dirs), 0)

    def test_find_rank_dirs_with_rank_dirs(self):
        """Test finding rank directories when they exist."""
        hash_dir = self.vllm_cache_root / "test_hash"
        hash_dir.mkdir()

        # Create rank directories with triton_cache subdirs
        rank1 = hash_dir / "rank0_0"
        rank2 = hash_dir / "rank1_0"
        rank1.mkdir()
        rank2.mkdir()

        triton_cache1 = rank1 / "triton_cache"
        triton_cache2 = rank2 / "triton_cache"
        triton_cache1.mkdir()
        triton_cache2.mkdir()

        repo = VllmCacheRepository(self.vllm_cache_root)
        rank_dirs = list(repo._find_rank_dirs(hash_dir))

        self.assertEqual(len(rank_dirs), 2)
        self.assertIn(triton_cache1, rank_dirs)
        self.assertIn(triton_cache2, rank_dirs)

    def test_find_rank_dirs_without_triton_cache(self):
        """Test finding rank directories when they don't have triton_cache subdirs."""
        hash_dir = self.vllm_cache_root / "test_hash"
        hash_dir.mkdir()

        # Create rank directories without triton_cache subdirs
        rank1 = hash_dir / "rank0_0"
        rank1.mkdir()

        repo = VllmCacheRepository(self.vllm_cache_root)
        rank_dirs = list(repo._find_rank_dirs(hash_dir))

        self.assertEqual(len(rank_dirs), 0)

    @patch('model_cache_manager.data.cache_repo.CacheRepository')
    def test_kernels_empty_structure(self, mock_cache_repo_class):
        """Test kernels method with empty vLLM structure."""
        mock_cache_repo = MagicMock()
        mock_cache_repo.kernels.return_value = []
        mock_cache_repo_class.return_value = mock_cache_repo

        repo = VllmCacheRepository(self.vllm_cache_root)
        kernels = list(repo.kernels())

        self.assertEqual(len(kernels), 0)

    @patch('model_cache_manager.data.cache_repo.iter_triton_kernels')
    def test_kernels_with_structure_and_kernels(self, mock_iter_triton_kernels):
        """Test kernels method with vLLM structure containing kernels."""
        # Create vLLM directory structure
        torch_compile_dir = self.vllm_cache_root / "torch_compile_cache"
        torch_compile_dir.mkdir()

        hash_dir = torch_compile_dir / "hash123abc"
        hash_dir.mkdir()

        rank_dir = hash_dir / "rank0_0"
        rank_dir.mkdir()

        triton_cache = rank_dir / "triton_cache"
        triton_cache.mkdir()

        # Mock the iter_triton_kernels function to return fake kernels
        mock_kernel1 = MagicMock(spec=Kernel)
        mock_kernel1.hash = "kernel_hash_1"
        mock_kernel1.name = "test_kernel_1"

        mock_kernel2 = MagicMock(spec=Kernel)
        mock_kernel2.hash = "kernel_hash_2"
        mock_kernel2.name = "test_kernel_2"

        mock_iter_triton_kernels.return_value = [mock_kernel1, mock_kernel2]

        repo = VllmCacheRepository(self.vllm_cache_root)
        kernels = list(repo.kernels())

        # Should have 2 kernels, each with vllm_hash and vllm_cache_root
        self.assertEqual(len(kernels), 2)

        vllm_hash, vllm_cache_root, kernel = kernels[0]
        self.assertEqual(vllm_hash, "hash123abc")
        self.assertEqual(vllm_cache_root, str(self.vllm_cache_root))
        self.assertEqual(kernel, mock_kernel1)

        vllm_hash, vllm_cache_root, kernel = kernels[1]
        self.assertEqual(vllm_hash, "hash123abc")
        self.assertEqual(vllm_cache_root, str(self.vllm_cache_root))
        self.assertEqual(kernel, mock_kernel2)

    @patch('model_cache_manager.data.cache_repo.iter_triton_kernels')
    def test_kernels_multiple_hash_dirs(self, mock_iter_triton_kernels):
        """Test kernels method with multiple hash directories."""
        # Create vLLM directory structure with multiple hash dirs
        torch_compile_dir = self.vllm_cache_root / "torch_compile_cache"
        torch_compile_dir.mkdir()

        hash_dir1 = torch_compile_dir / "hash123abc"
        hash_dir2 = torch_compile_dir / "hash456def"
        hash_dir1.mkdir()
        hash_dir2.mkdir()

        # Create rank dirs for each hash
        for hash_dir in [hash_dir1, hash_dir2]:
            rank_dir = hash_dir / "rank0_0"
            rank_dir.mkdir()
            triton_cache = rank_dir / "triton_cache"
            triton_cache.mkdir()

        # Mock kernels for each cache directory call
        def side_effect(triton_cache_path, plugins):
            if "hash123abc" in str(triton_cache_path):
                mock_kernel = MagicMock(spec=Kernel)
                mock_kernel.hash = f"kernel_from_hash123abc"
                return [mock_kernel]
            elif "hash456def" in str(triton_cache_path):
                mock_kernel = MagicMock(spec=Kernel)
                mock_kernel.hash = f"kernel_from_hash456def"
                return [mock_kernel]
            return []

        mock_iter_triton_kernels.side_effect = side_effect

        repo = VllmCacheRepository(self.vllm_cache_root)
        kernels = list(repo.kernels())

        # Should have kernels from both hash directories
        self.assertEqual(len(kernels), 2)

        vllm_hashes = [vllm_hash for vllm_hash, _, _ in kernels]
        self.assertIn("hash123abc", vllm_hashes)
        self.assertIn("hash456def", vllm_hashes)


if __name__ == "__main__":
    unittest.main()
