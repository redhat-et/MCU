"""
Strategy factory for creating appropriate strategy instances based on mode.
"""

# pylint: disable=relative-beyond-top-level
from ..utils.mcm_constants import MODE_VLLM, MODE_VLLM_LEGACY, MODE_HELION


def create_strategy(mode: str):
    """
    Factory function to create the appropriate strategy based on mode.

    Args:
        mode: The mode string (MODE_VLLM, MODE_VLLM_LEGACY, MODE_HELION, or default)

    Returns:
        An instance of the appropriate strategy class
    """
    # Import here to avoid circular dependencies
    # pylint: disable=import-outside-toplevel,relative-beyond-top-level
    from ..strategies.vllm_strategy import VllmStrategy
    from ..strategies.vllm_legacy_strategy import VllmLegacyStrategy
    from ..strategies.triton_strategy import TritonStrategy
    from ..strategies.helion_strategy import HelionStrategy

    if mode == MODE_VLLM:
        return VllmStrategy()
    if mode == MODE_VLLM_LEGACY:
        return VllmLegacyStrategy()
    if mode == MODE_HELION:
        return HelionStrategy()
    return TritonStrategy()
