"""
Unit tests for the HelionCacheRepository.
"""

import json
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil

from model_cache_manager.data.cache_repo import HelionCacheRepository, HELION_KERNEL_PREFIX
from model_cache_manager.models.kernel import Kernel


class TestHelionCacheRepository(unittest.TestCase):
    """Test suite for the HelionCacheRepository."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "helion_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_with_existing_directory(self):
        """Test initializing HelionCacheRepository with existing directory."""
        repo = HelionCacheRepository(self.cache_dir)
        self.assertEqual(repo.root, self.cache_dir)

    def test_init_with_nonexistent_directory(self):
        """Test initializing HelionCacheRepository with non-existent directory."""
        nonexistent = self.temp_dir / "nonexistent"
        with self.assertRaises(FileNotFoundError):
            HelionCacheRepository(nonexistent)

    def test_read_best_configs_empty(self):
        """Test reading best configs when none exist."""
        repo = HelionCacheRepository(self.cache_dir)
        # pylint: disable=protected-access
        configs = repo._read_best_configs()
        self.assertEqual(configs, {})

    def test_read_best_configs(self):
        """Test reading best_config files."""
        best_config_data = {
            "config": '{"block_sizes": [32]}',
            "backend_cache_key": "KERNEL_ABC",
        }
        (self.cache_dir / "abc123.best_config").write_text(
            json.dumps(best_config_data)
        )

        repo = HelionCacheRepository(self.cache_dir)
        # pylint: disable=protected-access
        configs = repo._read_best_configs()
        self.assertIn("KERNEL_ABC", configs)
        helion_hash, raw = configs["KERNEL_ABC"]
        self.assertEqual(helion_hash, "abc123")
        self.assertIn("backend_cache_key", raw)

    def test_read_best_configs_skips_invalid_json(self):
        """Test that invalid JSON best_config files are skipped."""
        (self.cache_dir / "bad.best_config").write_text("not json")
        (self.cache_dir / "good.best_config").write_text(
            json.dumps({"backend_cache_key": "KEY1"})
        )

        repo = HelionCacheRepository(self.cache_dir)
        # pylint: disable=protected-access
        configs = repo._read_best_configs()
        self.assertEqual(len(configs), 1)
        self.assertIn("KEY1", configs)

    def test_read_best_configs_skips_missing_backend_key(self):
        """Test that best_config without backend_cache_key is skipped."""
        (self.cache_dir / "nokey.best_config").write_text(
            json.dumps({"config": "test"})
        )

        repo = HelionCacheRepository(self.cache_dir)
        # pylint: disable=protected-access
        configs = repo._read_best_configs()
        self.assertEqual(configs, {})

    def test_kernels_empty(self):
        """Test kernels with no triton directory."""
        repo = HelionCacheRepository(self.cache_dir)
        kernels = list(repo.kernels())
        self.assertEqual(kernels, [])

    def test_kernels_yields_helion_kernels(self):
        """Test that kernels yields only _helion_ prefixed kernels."""
        triton_dir = self.cache_dir / "triton" / "0"
        triton_dir.mkdir(parents=True)
        (triton_dir / "KERNEL_A").mkdir()

        best_config_data = {"backend_cache_key": "KERNEL_A"}
        (self.cache_dir / "hash1.best_config").write_text(
            json.dumps(best_config_data)
        )

        repo = HelionCacheRepository(self.cache_dir)
        with patch(
            "model_cache_manager.data.cache_repo.iter_triton_kernels"
        ) as mock_iter:
            helion_kernel = MagicMock(spec=Kernel)
            helion_kernel.name = "_helion_add"
            helion_kernel.hash = "KERNEL_A"

            regular_kernel = MagicMock(spec=Kernel)
            regular_kernel.name = "triton_matmul"
            regular_kernel.hash = "KERNEL_B"

            mock_iter.return_value = [helion_kernel, regular_kernel]

            results = list(repo.kernels())

        self.assertEqual(len(results), 1)
        _cache_dir, triton_key, helion_hash, best_config, is_best, kernel = results[0]
        self.assertEqual(kernel.name, "_helion_add")
        self.assertEqual(triton_key, "KERNEL_A")
        self.assertEqual(helion_hash, "hash1")
        self.assertTrue(is_best)

    def test_kernels_is_best_false_when_no_match(self):
        """Test that is_best is False when kernel hash doesn't match any best_config."""
        triton_dir = self.cache_dir / "triton" / "0"
        triton_dir.mkdir(parents=True)

        best_config_data = {"backend_cache_key": "OTHER_KEY"}
        (self.cache_dir / "hash1.best_config").write_text(
            json.dumps(best_config_data)
        )

        repo = HelionCacheRepository(self.cache_dir)
        with patch(
            "model_cache_manager.data.cache_repo.iter_triton_kernels"
        ) as mock_iter:
            kernel = MagicMock(spec=Kernel)
            kernel.name = "_helion_add"
            kernel.hash = "UNMATCHED_KEY"
            mock_iter.return_value = [kernel]

            results = list(repo.kernels())

        self.assertEqual(len(results), 1)
        _, _, helion_hash, best_config, is_best, _ = results[0]
        self.assertIsNone(helion_hash)
        self.assertIsNone(best_config)
        self.assertFalse(is_best)

    def test_kernels_skips_kernel_with_no_name(self):
        """Test that kernels with no name are skipped."""
        (self.cache_dir / "triton" / "0").mkdir(parents=True)

        repo = HelionCacheRepository(self.cache_dir)
        with patch(
            "model_cache_manager.data.cache_repo.iter_triton_kernels"
        ) as mock_iter:
            kernel = MagicMock(spec=Kernel)
            kernel.name = None
            kernel.hash = "K1"
            mock_iter.return_value = [kernel]

            results = list(repo.kernels())

        self.assertEqual(results, [])

    def test_kernels_multiple_best_configs(self):
        """Test with multiple best_config files mapping to different kernels."""
        (self.cache_dir / "triton" / "0").mkdir(parents=True)

        (self.cache_dir / "h1.best_config").write_text(
            json.dumps({"backend_cache_key": "K1"})
        )
        (self.cache_dir / "h2.best_config").write_text(
            json.dumps({"backend_cache_key": "K2"})
        )

        repo = HelionCacheRepository(self.cache_dir)
        with patch(
            "model_cache_manager.data.cache_repo.iter_triton_kernels"
        ) as mock_iter:
            k1 = MagicMock(spec=Kernel)
            k1.name = "_helion_op1"
            k1.hash = "K1"

            k2 = MagicMock(spec=Kernel)
            k2.name = "_helion_op2"
            k2.hash = "K2"

            mock_iter.return_value = [k1, k2]
            results = list(repo.kernels())

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0][4])  # is_best
        self.assertTrue(results[1][4])  # is_best

    def test_helion_kernel_prefix_constant(self):
        """Test that the prefix constant matches expected value."""
        self.assertEqual(HELION_KERNEL_PREFIX, "_helion_")


if __name__ == "__main__":
    unittest.main()
