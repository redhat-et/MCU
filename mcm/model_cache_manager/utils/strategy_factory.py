"""
Strategy factory for creating appropriate strategy instances based on mode.
"""

from ..utils.mcm_constants import MODE_VLLM, MODE_VLLM_LEGACY


def create_strategy(mode: str):
    """
    Factory function to create the appropriate strategy based on mode.

    Args:
        mode: The mode string (MODE_VLLM, MODE_VLLM_LEGACY, or default)

    Returns:
        An instance of the appropriate strategy class
    """
    # Import here to avoid circular dependencies
    from ..strategies.vllm_strategy import VllmStrategy
    from ..strategies.vllm_legacy_strategy import VllmLegacyStrategy
    from ..strategies.triton_strategy import TritonStrategy

    if mode == MODE_VLLM:
        return VllmStrategy()
    elif mode == MODE_VLLM_LEGACY:
        return VllmLegacyStrategy()
    else:
        return TritonStrategy()