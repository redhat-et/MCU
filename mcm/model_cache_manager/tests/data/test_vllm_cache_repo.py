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
        artifact_kernels = list(repo._find_artifact_kernels(rank_dir, "rank_0_0", "test_vllm_hash"))
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

            artifact_kernels = list(
                repo._find_artifact_kernels(rank_dir, "rank_0_0", "test_vllm_hash")
            )
        # pylint: enable=protected-access

        self.assertEqual(len(artifact_kernels), 1)
        artifact_compile_range, best_config, triton_subpath, kernel = artifact_kernels[0]
        self.assertEqual(artifact_compile_range, "artifact_compile_range_0")
        self.assertEqual(best_config, '{"config": "test"}')
        self.assertIsNone(triton_subpath)  # Direct triton/ path
        self.assertEqual(kernel.hash, "test_kernel")

    def test_find_artifact_kernels_without_backbone(self):
        """Test finding artifact kernels when backbone dir doesn't exist."""
        rank_dir = self.cache_dir / "rank_0_0"
        rank_dir.mkdir()
        # No backbone directory created

        repo = VllmCacheRepository(self.cache_dir)
        # Testing protected method is ok in unit tests
        # pylint: disable=protected-access
        artifact_kernels = list(repo._find_artifact_kernels(rank_dir, "rank_0_0", "test_vllm_hash"))
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
        #  best_config, triton_subpath, and kernel
        self.assertEqual(len(kernels), 2)

        (
            vllm_hash, cache_dir, rank_x_y, artifact_compile_range,
            best_config, triton_subpath, kernel
        ) = kernels[0]
        self.assertEqual(vllm_hash, "hash123abc")
        self.assertEqual(cache_dir, str(self.cache_dir))
        self.assertEqual(rank_x_y, "rank_0_0")
        self.assertEqual(artifact_compile_range, "artifact_compile_range_0")
        self.assertIsNone(best_config)  # No best_config file created
        self.assertIsNone(triton_subpath)  # Direct triton/ path
        self.assertEqual(kernel, mock_kernel1)

        (
            vllm_hash, cache_dir, rank_x_y, artifact_compile_range,
            best_config, triton_subpath, kernel
        ) = kernels[1]
        self.assertEqual(vllm_hash, "hash123abc")
        self.assertEqual(cache_dir, str(self.cache_dir))
        self.assertEqual(rank_x_y, "rank_0_0")
        self.assertEqual(artifact_compile_range, "artifact_compile_range_0")
        self.assertIsNone(best_config)  # No best_config file created
        self.assertIsNone(triton_subpath)  # Direct triton/ path
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

        vllm_hashes = [vllm_hash for vllm_hash, _, _, _, _, _, _ in kernels]
        self.assertIn("hash123abc", vllm_hashes)
        self.assertIn("hash456def", vllm_hashes)


class TestVllmCacheRepositoryBinaryArtifacts(unittest.TestCase):
    """Test suite for VllmCacheRepository with binary artifacts."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "vllm_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_binary_artifact_structure(self) -> Path:
        """Create binary artifact directory structure for testing."""
        rank_dir = self.cache_dir / "rank_0_0"
        rank_dir.mkdir()
        backbone_dir = rank_dir / "backbone"
        backbone_dir.mkdir()
        artifact_dir = backbone_dir / "artifact_compile_range_0"
        artifact_dir.mkdir()
        binary_file = artifact_dir / "artifact_compile_range_0"
        binary_file.touch()
        return artifact_dir

    def _create_temp_extraction_dir(self) -> Path:
        """Create temp extraction directory with triton structure."""
        temp_extract_dir = self.temp_dir / "temp_extract"
        temp_extract_dir.mkdir(exist_ok=True)
        temp_triton_dir = temp_extract_dir / "triton"
        temp_triton_dir.mkdir(exist_ok=True)
        (temp_triton_dir / "kernel_hash").mkdir(exist_ok=True)
        return temp_extract_dir

    def _create_vllm_structure(self) -> Path:
        """Create vLLM directory structure and return backbone_dir."""
        torch_compile_dir = self.cache_dir / "torch_compile_cache"
        torch_compile_dir.mkdir(exist_ok=True)
        hash_dir = torch_compile_dir / "hash123abc"
        hash_dir.mkdir(exist_ok=True)
        rank_dir = hash_dir / "rank_0_0"
        rank_dir.mkdir(exist_ok=True)
        backbone_dir = rank_dir / "backbone"
        backbone_dir.mkdir(exist_ok=True)
        return backbone_dir

    @patch("model_cache_manager.data.binary_artifact_extractor.TemporaryExtractedArtifact")
    @patch("model_cache_manager.data.cache_repo.iter_triton_kernels")
    def test_process_artifact_dir_with_binary_artifact(
        self, mock_iter_triton, mock_temp_extracted
    ):
        """Test _process_artifact_dir handles binary artifacts via extraction."""
        # Create binary artifact structure
        artifact_dir = self._create_binary_artifact_structure()

        # Create temp extraction directory that will be returned
        temp_extract_dir = self._create_temp_extraction_dir()

        # Mock the context manager to return temp directory
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=temp_extract_dir)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_temp_extracted.return_value = mock_context

        # Mock kernel iteration
        mock_kernel = MagicMock(spec=Kernel)
        mock_kernel.hash = "binary_kernel_hash"
        mock_iter_triton.return_value = [mock_kernel]

        repo = VllmCacheRepository(self.cache_dir)
        # pylint: disable=protected-access
        plugins = {}
        kernels = list(repo._process_artifact_dir(
            artifact_dir, plugins, "test_vllm_hash", "rank_0_0"
        ))
        # pylint: enable=protected-access

        # Verify extraction context manager was called
        mock_temp_extracted.assert_called_once_with(
            artifact_dir, "test_vllm_hash", "rank_0_0"
        )

        # Verify we got the kernel
        self.assertEqual(len(kernels), 1)
        artifact_name, _best_config, triton_subpath, kernel = kernels[0]
        self.assertEqual(artifact_name, "artifact_compile_range_0")
        self.assertIsNone(triton_subpath)
        self.assertEqual(kernel.hash, "binary_kernel_hash")

    @patch("model_cache_manager.data.binary_artifact_extractor.TemporaryExtractedArtifact")
    def test_process_artifact_dir_binary_extraction_fails(self, mock_temp_extracted):
        """Test _process_artifact_dir handles extraction failures gracefully."""
        artifact_dir = self._create_binary_artifact_structure()

        # Mock extraction failure
        # pylint: disable=import-outside-toplevel
        from model_cache_manager.data.binary_artifact_extractor import (
            BinaryArtifactExtractionError,
        )
        mock_temp_extracted.return_value.__enter__.side_effect = (
            BinaryArtifactExtractionError("Extraction failed")
        )

        repo = VllmCacheRepository(self.cache_dir)
        # pylint: disable=protected-access
        plugins = {}
        kernels = list(repo._process_artifact_dir(
            artifact_dir, plugins, "test_vllm_hash", "rank_0_0"
        ))
        # pylint: enable=protected-access

        # Should return empty list and not raise
        self.assertEqual(len(kernels), 0)

    @patch("model_cache_manager.data.binary_artifact_extractor.TemporaryExtractedArtifact")
    @patch("model_cache_manager.data.cache_repo.iter_triton_kernels")
    def test_kernels_with_mixed_binary_and_unpacked_artifacts(
        self, mock_iter_triton, mock_temp_extracted
    ):  # pylint: disable=too-many-locals
        """Test kernels method with mix of binary and unpacked artifacts."""
        # Create vLLM structure with mixed artifact types
        backbone_dir = self._create_vllm_structure()
        temp_extract_dir = self._create_temp_extraction_dir()
        # Add kernel2 directory for the binary artifact extraction
        (temp_extract_dir / "triton" / "kernel2").mkdir(exist_ok=True)

        # Create unpacked artifact (with triton/ dir)
        unpacked_artifact = backbone_dir / "artifact_compile_range_0"
        unpacked_artifact.mkdir()
        unpacked_triton = unpacked_artifact / "triton"
        unpacked_triton.mkdir()
        (unpacked_triton / "kernel1").mkdir()

        # Create binary artifact (no triton/ dir, has binary file)
        binary_artifact = backbone_dir / "artifact_compile_range_1"
        binary_artifact.mkdir()
        (binary_artifact / "artifact_compile_range_1").touch()

        # Mock context manager - returns original for unpacked, temp for binary
        def context_side_effect(artifact_dir, _vllm_hash, _rank_name):
            mock_context = MagicMock()
            if "artifact_compile_range_0" in str(artifact_dir):
                mock_context.__enter__ = MagicMock(return_value=artifact_dir)
            else:
                mock_context.__enter__ = MagicMock(return_value=temp_extract_dir)
            mock_context.__exit__ = MagicMock(return_value=False)
            return mock_context

        mock_temp_extracted.side_effect = context_side_effect

        # Mock kernels for each artifact
        mock_kernel1, mock_kernel2 = MagicMock(spec=Kernel), MagicMock(spec=Kernel)
        mock_kernel1.hash, mock_kernel2.hash = "unpacked_kernel", "binary_kernel"

        def iter_side_effect(kernel_dir, _plugins):
            if "kernel1" in str(kernel_dir):
                return [mock_kernel1]
            if "kernel2" in str(kernel_dir):
                return [mock_kernel2]
            return []

        mock_iter_triton.side_effect = iter_side_effect

        repo = VllmCacheRepository(self.cache_dir)
        kernels = list(repo.kernels())

        # Should have kernels from both artifacts
        self.assertEqual(len(kernels), 2)

        # Verify both artifacts were processed
        artifact_names = [
            artifact_compile_range
            for _, _, _, artifact_compile_range, _, _, _ in kernels
        ]
        self.assertIn("artifact_compile_range_0", artifact_names)
        self.assertIn("artifact_compile_range_1", artifact_names)

        # Verify kernel hashes
        kernel_hashes = [k.hash for _, _, _, _, _, _, k in kernels]
        self.assertIn("unpacked_kernel", kernel_hashes)
        self.assertIn("binary_kernel", kernel_hashes)


if __name__ == "__main__":
    unittest.main()
