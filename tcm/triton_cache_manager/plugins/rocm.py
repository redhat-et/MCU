from .base import KernelBackendPlugin


class RocmPlugin(KernelBackendPlugin):
    backend = "rocm"

    def relevant_extensions(self):
        return {".amdgcn": "amdgcn", ".hsaco": "hsaco"}
