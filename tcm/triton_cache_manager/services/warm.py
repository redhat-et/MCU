"""
Service for warming the vLLM cache using a container.
"""

import logging
import subprocess
import shutil
import os
from datetime import datetime
from pathlib import Path

from ..utils.paths import get_cache_dir
from ..utils.utils import format_size

log = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class WarmupService:
    """
    Manages the cache warming process by running a containerized script
    and packaging the resulting cache directory.
    """

    def __init__(
        self, model: str, hface_secret: str, vllm_cache_dir: str, host_cache_dir: str
    ):
        """
        Initializes the WarmupService.
        """
        self.vllm_cache_dir = vllm_cache_dir or get_cache_dir()
        self.hface_secret = hface_secret
        self.vllm_cache_dir = vllm_cache_dir
        self.host_cache_dir = host_cache_dir
        self.model = model

    def warmup(self, image: str, output_file: Path, tarball: bool, rocm: bool) -> bool:
        """
        Executes the full cache warming and packaging workflow.

        This method runs a Podman container to generate cache files and,
        upon success, optionally packages the cache directory into a tarball.

        Args:
            image: The container image to use for warming the cache.
            output_file: The path to save the final packaged cache file.
            tarball: True if user wants to compress the warmed cache
            rocm: True if user uses a rocm GPU

        Returns:
            True if the process completes successfully, False otherwise.
        """
        log.info("Starting cache warmup process with image: '%s'", image)

        if not self._run_container(image, rocm):
            log.error("Cache warming container failed to execute successfully.")
            return False

        if tarball:
            log.info("Container completed successfully. Now packaging the cache...")
            try:
                self._package_cache(output_file)
                final_size = output_file.stat().st_size
                log.info(
                    "Cache packaged successfully to '%s' (Size: %s)",
                    output_file,
                    format_size(final_size),
                )
                return True
            except (OSError, shutil.Error) as e:
                log.error("Failed to package the cache directory: %s", e, exc_info=True)
                return False
        return True

    def _create_date_time_dir(self) -> str:
        """Create a date time directory for the warmed cache"""
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        dir_name = f"{self.host_cache_dir}/cache_{timestamp}"
        try:
            os.makedirs(dir_name)
            print(f"Directory '{dir_name}' created successfully.")
        except FileExistsError:
            log.error("Directory %s already exists.", dir_name)
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error("Error creating directory: %s", e)

        return dir_name

    def _run_container(self, image: str, rocm: bool) -> bool:
        """
        Runs a Podman container with the cache directory mounted as a volume.

        Args:
            image: The name of the container image to run.
            rocm: True if user uses a rocm GPU
        Returns:
            True if the container exits with a status code of 0, False otherwise.
        """

        env = {
            "MODEL_NAME": self.model,
            "VLLM_CACHE_ROOT": self.vllm_cache_dir,
        }
        if self.hface_secret:
            env["HF_TOKEN"] = self.hface_secret

        env_flags = [
            part for key, val in env.items() for part in ("-e", f"{key}={val}")
        ]

        host_cache_dir = self._create_date_time_dir()
        if rocm:
            backend_cmd = [
                "--ipc=host",
                "--network=host",
                "--cap-add=CAP_SYS_ADMIN",
                "--device=/dev/kfd",
                "--device=/dev/dri",
                "--device=/dev/mem",
                "--group-add",
                "render",
                "--cap-add=SYS_PTRACE",
                "--security-opt",
                "seccomp=unconfined",
            ]
        else:
            backend_cmd = [
                "--device",
                "nvidia.com/gpu=all",
            ]

        volume_mount = f"{host_cache_dir}:{self.vllm_cache_dir}:Z,rw"
        command = [
            "podman",
            "run",
            "--privileged",
            "-it",
            *backend_cmd,
            *env_flags,
            "--volume",
            volume_mount,
            image,
        ]

        log.debug("Executing Podman command: %s", " ".join(command))
        try:
            result = subprocess.run(
                command, check=True, capture_output=False, text=True, encoding="utf-8"
            )
            log.debug("Container stdout:\n%s", result.stdout)
            log.debug("Container stderr:\n%s", result.stderr)
            log.info("Podman container ran and exited successfully.")
            return True
        except FileNotFoundError:
            log.error(
                "The 'podman' command was not found. "
                "Please ensure Podman is installed and accessible in your system's PATH."
            )
            return False
        except subprocess.CalledProcessError as e:
            log.error(
                "Podman container exited with a non-zero status code: %d.", e.returncode
            )
            log.error("Container stdout:\n%s", e.stdout)
            log.error("Container stderr:\n%s", e.stderr)
            return False

    def _package_cache(self, output_file: Path):
        """
        Creates a gzipped tarball from the cache directory.

        Args:
            output_file: The target path for the archive, including the extension.

        Raises:
            shutil.Error: If there is an error during the archive creation.
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        base_name = output_file.with_suffix("").with_suffix("")
        archive_format = "gztar"

        log.info(
            "Creating '%s' archive from directory '%s'...",
            archive_format,
            self.host_cache_dir,
        )
        shutil.make_archive(
            base_name=str(base_name),
            format=archive_format,
            root_dir=self.host_cache_dir,
        )
