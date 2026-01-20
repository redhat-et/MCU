"""
Shared constants for strategy configurations.
"""

# Common primary key fields for vLLM strategies
VLLM_COMMON_PRIMARY_FIELDS = [
    "cache_dir",
    "vllm_hash",
    "triton_cache_key",
    "rank_x_y",
]

# Extended primary key fields for new vLLM strategy
VLLM_EXTENDED_PRIMARY_FIELDS = VLLM_COMMON_PRIMARY_FIELDS + ["artifact_compile_range"]

# Hash field name for vLLM strategies
VLLM_HASH_FIELD = "triton_cache_key"
