from .base import KernelBackendPlugin


class CudaPlugin(KernelBackendPlugin):
    backend = "cuda"

    def relevant_extensions(self):
        return {".ptx": "ptx", ".cubin": "cubin"}
