from typing import Protocol, Dict


class KernelBackendPlugin(Protocol):
    backend: str

    def relevant_extensions(self) -> Dict[str, str]: ...
def discover_plugins():
    from .cuda import CudaPlugin
    from .rocm import RocmPlugin

    return [CudaPlugin(), RocmPlugin()]
