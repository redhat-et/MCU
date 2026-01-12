"""
Unit tests for the VllmCacheRepository.
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil

from model_cache_manager.data.cache_repo import VllmCacheRepository
from model_cache_manager.models.kernel import Kernel


class TestVllmCacheRepository(unittest.TestCase):
    """Test suite for the VllmCacheRepository."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "vllm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_with_existing_directory(self):
        """Test initializing VllmCacheRepository with existing directory."""
        repo = VllmCacheRepository(self.cache_dir)
        self.assertEqual(repo.root, self.cache_dir)

    def test_init_with_nonexistent_directory(self):
        """Test initializing VllmCacheRepository with non-existent directory."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        with self.assertRaises(FileNotFoundError):
            VllmCacheRepository(nonexistent_dir)

    def test_init_with_default_directory(self):
        """Test initializing VllmCacheRepository with default directory."""
        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = self.temp_dir
            default_vllm_cache = self.temp_dir / ".cache" / "vllm"
            default_vllm_cache.mkdir(parents=True, exist_ok=True)

            repo = VllmCacheRepository()
            self.assertEqual(repo.root, default_vllm_cache)

    def test_find_torch_compile_cache_dirs_empty(self):
        """Test finding torch compile cache directories when none exist."""
        repo = VllmCacheRepository(self.cache_dir)
        # Testing protected method is ok in unit tests
        # pylint: disable=protected-access
        dirs = list(repo._find_torch_compile_cache_dirs())
        # pylint: enable=protected-access
        self.assertEqual(len(dirs), 0)

    def test_find_torch_compile_cache_dirs_with_dirs(self):
        """Test finding torch compile cache directories when they exist."""
        # Create mock vLLM structure
        torch_compile_dir = self.cache_dir / "torch_compile_cache"
        torch_compile_dir.mkdir()

        hash_dir1 = torch_compile_dir / "hash123abc"
        hash_dir2 = torch_compile_dir / "hash456def"
        hash_dir1.mkdir()
        hash_dir2.mkdir()

        repo = VllmCacheRepository(self.cache_dir)
        # Testing protected method is ok in unit tests
        # pylint: disable=protected-access
        dirs = list(repo._find_torch_compile_cache_dirs())
        # pylint: enable=protected-access

        self.assertEqual(len(dirs), 2)
        hash_names = [hash_name for hash_name, _ in dirs]
        self.assertIn("hash123abc", hash_names)
        self.assertIn("hash456def", hash_names)

    def test_find_artifact_kernels_empty(self):
        """Test finding artifact kernels when none exist."""
        rank_dir = self.cache_dir / "test_rank"
        rank_dir.mkdir()

        repo = VllmCacheRepository(self.cache_dir)
        # Testing protected method is ok in unit tests
        # pylint: disable=protected-access
        artifact_kernels = list(repo._find_artifact_kernels(rank_dir, "rank_0_0"))
        # pylint: enable=protected-access
        self.assertEqual(len(artifact_kernels), 0)

    def test_find_artifact_kernels_with_structure(self):
        """Test finding artifact kernels when they exist."""
        rank_dir = self.cache_dir / "rank_0_0"
        rank_dir.mkdir()

        # Create new vLLM structure with backbone and artifact_compile_range dirs
        backbone_dir = rank_dir / "backbone"
        backbone_dir.mkdir()

        artifact_dir = backbone_dir / "artifact_compile_range_0"
        artifact_dir.mkdir()

        # Create best config file
        best_config_file = artifact_dir / "test.best_config"
        best_config_file.write_text('{"config": "test"}')

        # Create triton directory structure
        triton_dir = artifact_dir / "triton"
        triton_dir.mkdir()

        kernel_dir = triton_dir / "test_kernel_hash"
        kernel_dir.mkdir()

        repo = VllmCacheRepository(self.cache_dir)
        # Testing protected method is ok in unit tests
        # pylint: disable=protected-access
        with patch(
            "model_cache_manager.data.cache_repo.iter_triton_kernels"
        ) as mock_iter:
            mock_kernel = MagicMock(spec=Kernel)
            mock_kernel.hash = "test_kernel"
            mock_iter.return_value = [mock_kernel]

            artifact_kernels = list(repo._find_artifact_kernels(rank_dir, "rank_0_0"))
        # pylint: enable=protected-access

        self.assertEqual(len(artifact_kernels), 1)
        artifact_compile_range, best_config, kernel = artifact_kernels[0]
        self.assertEqual(artifact_compile_range, "artifact_compile_range_0")
        self.assertEqual(best_config, '{"config": "test"}')
        self.assertEqual(kernel.hash, "test_kernel")

    def test_find_artifact_kernels_without_backbone(self):
        """Test finding artifact kernels when backbone dir doesn't exist."""
        rank_dir = self.cache_dir / "rank_0_0"
        rank_dir.mkdir()
        # No backbone directory created

        repo = VllmCacheRepository(self.cache_dir)
        # Testing protected method is ok in unit tests
        # pylint: disable=protected-access
        artifact_kernels = list(repo._find_artifact_kernels(rank_dir, "rank_0_0"))
        # pylint: enable=protected-access

        self.assertEqual(len(artifact_kernels), 0)

    @patch("model_cache_manager.data.cache_repo.CacheRepository")
    def test_kernels_empty_structure(self, mock_cache_repo_class):
        """Test kernels method with empty vLLM structure."""
        mock_cache_repo = MagicMock()
        mock_cache_repo.kernels.return_value = []
        mock_cache_repo_class.return_value = mock_cache_repo

        repo = VllmCacheRepository(self.cache_dir)
        kernels = list(repo.kernels())

        self.assertEqual(len(kernels), 0)

    # pylint: disable=too-many-locals
    @patch("model_cache_manager.data.cache_repo.iter_triton_kernels")
    def test_kernels_with_structure_and_kernels(self, mock_iter_triton_kernels):
        """Test kernels method with vLLM structure containing kernels."""
        # Create new vLLM directory structure
        torch_compile_dir = self.cache_dir / "torch_compile_cache"
        torch_compile_dir.mkdir()

        hash_dir = torch_compile_dir / "hash123abc"
        hash_dir.mkdir()

        rank_dir = hash_dir / "rank_0_0"
        rank_dir.mkdir()

        # Create new structure: backbone/artifact_compile_range_0/triton/kernel_dir
        backbone_dir = rank_dir / "backbone"
        backbone_dir.mkdir()

        artifact_dir = backbone_dir / "artifact_compile_range_0"
        artifact_dir.mkdir()

        triton_dir = artifact_dir / "triton"
        triton_dir.mkdir()

        kernel_dir = triton_dir / "test_kernel_hash"
        kernel_dir.mkdir()

        # Mock the iter_triton_kernels function to return fake kernels
        mock_kernel1 = MagicMock(spec=Kernel)
        mock_kernel1.hash = "kernel_hash_1"
        mock_kernel1.name = "test_kernel_1"

        mock_kernel2 = MagicMock(spec=Kernel)
        mock_kernel2.hash = "kernel_hash_2"
        mock_kernel2.name = "test_kernel_2"

        mock_iter_triton_kernels.return_value = [mock_kernel1, mock_kernel2]

        repo = VllmCacheRepository(self.cache_dir)
        kernels = list(repo.kernels())

        # Should have 2 kernels, each with vllm_hash, cache_dir, rank_x_y, artifact_compile_range,
        #  best_config, and kernel
        self.assertEqual(len(kernels), 2)

        vllm_hash, cache_dir, rank_x_y, artifact_compile_range, best_config, kernel = kernels[0]
        self.assertEqual(vllm_hash, "hash123abc")
        self.assertEqual(cache_dir, str(self.cache_dir))
        self.assertEqual(rank_x_y, "rank_0_0")
        self.assertEqual(artifact_compile_range, "artifact_compile_range_0")
        self.assertIsNone(best_config)  # No best_config file created
        self.assertEqual(kernel, mock_kernel1)

        vllm_hash, cache_dir, rank_x_y, artifact_compile_range, best_config, kernel = kernels[1]
        self.assertEqual(vllm_hash, "hash123abc")
        self.assertEqual(cache_dir, str(self.cache_dir))
        self.assertEqual(rank_x_y, "rank_0_0")
        self.assertEqual(artifact_compile_range, "artifact_compile_range_0")
        self.assertIsNone(best_config)  # No best_config file created
        self.assertEqual(kernel, mock_kernel2)

    @patch("model_cache_manager.data.cache_repo.iter_triton_kernels")
    def test_kernels_multiple_hash_dirs(self, mock_iter_triton_kernels):
        """Test kernels method with multiple hash directories."""
        # Create new vLLM directory structure with multiple hash dirs
        torch_compile_dir = self.cache_dir / "torch_compile_cache"
        torch_compile_dir.mkdir()

        hash_dir1 = torch_compile_dir / "hash123abc"
        hash_dir2 = torch_compile_dir / "hash456def"
        hash_dir1.mkdir()
        hash_dir2.mkdir()

        # Create new structure for each hash
        for hash_dir in [hash_dir1, hash_dir2]:
            rank_dir = hash_dir / "rank_0_0"
            rank_dir.mkdir()

            backbone_dir = rank_dir / "backbone"
            backbone_dir.mkdir()

            artifact_dir = backbone_dir / "artifact_compile_range_0"
            artifact_dir.mkdir()

            triton_dir = artifact_dir / "triton"
            triton_dir.mkdir()

            kernel_dir = triton_dir / "test_kernel_hash"
            kernel_dir.mkdir()

        # Mock kernels for each cache directory call
        def side_effect(triton_cache_path, plugins):  # pylint: disable=unused-argument
            if "hash123abc" in str(triton_cache_path):
                mock_kernel = MagicMock(spec=Kernel)
                mock_kernel.hash = "kernel_from_hash123abc"
                return [mock_kernel]
            if "hash456def" in str(triton_cache_path):
                mock_kernel = MagicMock(spec=Kernel)
                mock_kernel.hash = "kernel_from_hash456def"
                return [mock_kernel]
            return []

        mock_iter_triton_kernels.side_effect = side_effect

        repo = VllmCacheRepository(self.cache_dir)
        kernels = list(repo.kernels())

        # Should have kernels from both hash directories
        self.assertEqual(len(kernels), 2)

        vllm_hashes = [vllm_hash for vllm_hash, _, _, _, _, _ in kernels]
        self.assertIn("hash123abc", vllm_hashes)
        self.assertIn("hash456def", vllm_hashes)


if __name__ == "__main__":
    unittest.main()
