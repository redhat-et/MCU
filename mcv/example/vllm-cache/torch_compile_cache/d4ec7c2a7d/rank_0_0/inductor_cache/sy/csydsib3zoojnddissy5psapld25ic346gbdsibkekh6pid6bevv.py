# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/vllm/.cache/vllm/torch_compile_cache/d4ec7c2a7d/rank_0_0/inductor_cache/mk/cmk34tieuty2rpoli3ynf2ihs3dcfx6o5sv2l7rkt5vj44p2tk74.py
# Topologically Sorted Source Nodes: [long, embedding], Original ATen: [aten._to_copy, aten.embedding]
# Source node to ATen node mapping:
#   embedding => embedding
#   long => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.int64), kwargs = {})
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %convert_element_type), kwargs = {})
triton_poi_fused__to_copy_embedding_0 = async_compile.triton('triton_poi_fused__to_copy_embedding_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=104, cc='gfx90a', major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_embedding_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5F6849C846FE45386D7FD4995E383DFF233E57C8430A3EA7CA224D4D096E26A5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_embedding_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 1024
    x0 = (xindex % 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([XBLOCK], 151936, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 151936")
    tmp7 = tl.load(in_ptr1 + (x0 + 1024*tmp5), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: /home/vllm/.cache/vllm/torch_compile_cache/d4ec7c2a7d/rank_0_0/inductor_cache/47/c47v2of4lrhyox6un5t4z2l3xrtqxchprnn3ers2j65b5w64ghqu.py
# Topologically Sorted Source Nodes: [cat, cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
#   cat_2 => cat_1
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_19, %add_75], -1), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sub_36, %add_137], -1), kwargs = {})
triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=104, cc='gfx90a', major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '5F6849C846FE45386D7FD4995E383DFF233E57C8430A3EA7CA224D4D096E26A5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 16)
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp4.to(tl.int1)
    tmp6 = tl.load(in_ptr0 + (64*x1 + 3072*x2 + (x0)), xmask & tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x2), xmask & tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full([XBLOCK], 32768, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tl.broadcast_to(tmp11, [XBLOCK])) & (tl.broadcast_to(tmp11, [XBLOCK]) < 32768)) | ~(xmask & tmp5), "index out of bounds: 0 <= tl.broadcast_to(tmp11, [XBLOCK]) < 32768")
    tmp13 = tl.load(in_ptr2 + (64*tmp11 + (x0)), xmask & tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tmp6 * tmp13
    tmp15 = tl.load(in_ptr0 + (32 + 64*x1 + 3072*x2 + (x0)), xmask & tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr2 + (32 + 64*tmp11 + (x0)), xmask & tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 - tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tmp21 = tmp0 >= tmp3
    tmp22 = tl.full([1], 64, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21.to(tl.int1)
    tmp25 = tl.load(in_ptr0 + (32 + 64*x1 + 3072*x2 + ((-32) + x0)), xmask & tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (x2), xmask & tmp24, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.full([XBLOCK], 32768, tl.int32)
    tmp28 = tmp26 + tmp27
    tmp29 = tmp26 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp26)
    tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < 32768)) | ~(xmask & tmp24), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < 32768")
    tmp32 = tl.load(in_ptr2 + (64*tmp30 + ((-32) + x0)), xmask & tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tmp25 * tmp32
    tmp34 = tl.load(in_ptr0 + (64*x1 + 3072*x2 + ((-32) + x0)), xmask & tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.load(in_ptr2 + (32 + 64*tmp30 + ((-32) + x0)), xmask & tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp33 + tmp36
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp24, tmp37, tmp38)
    tmp40 = tl.where(tmp4, tmp20, tmp39)
    tmp41 = tl.load(in_ptr0 + (1024 + 64*x1 + 3072*x2 + (x0)), xmask & tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tmp41 * tmp13
    tmp43 = tl.load(in_ptr0 + (1056 + 64*x1 + 3072*x2 + (x0)), xmask & tmp5, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp44 = tmp43 * tmp16
    tmp45 = tmp42 - tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp5, tmp45, tmp46)
    tmp48 = tl.load(in_ptr0 + (1056 + 64*x1 + 3072*x2 + ((-32) + x0)), xmask & tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp49 = tmp48 * tmp32
    tmp50 = tl.load(in_ptr0 + (1024 + 64*x1 + 3072*x2 + ((-32) + x0)), xmask & tmp24, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp51 = tmp50 * tmp35
    tmp52 = tmp49 + tmp51
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp24, tmp52, tmp53)
    tmp55 = tl.where(tmp4, tmp47, tmp54)
    tl.store(out_ptr0 + (x4), tmp40, xmask)
    tl.store(out_ptr1 + (x4), tmp55, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
    args.clear()
    s0 = arg1_1
    assert_size_stride(arg0_1, (s0, ), (1, ))
    assert_size_stride(arg2_1, (151936, 1024), (1024, 1))
    assert_size_stride(arg3_1, (1024, ), (1, ))
    assert_size_stride(arg4_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg5_1, (3072, ), (1, ))
    assert_size_stride(arg6_1, (s0, ), (1, ))
    assert_size_stride(arg7_1, (32768, 64), (64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s0, 1024), (1024, 1), torch.bfloat16)
        buf1 = empty_strided_cuda((s0, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [long, embedding], Original ATen: [aten._to_copy, aten.embedding]
        triton_poi_fused__to_copy_embedding_0_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_embedding_0.run(arg0_1, arg2_1, buf1, triton_poi_fused__to_copy_embedding_0_xnumel, stream=stream0)
        del arg0_1
        del arg2_1
        # Topologically Sorted Source Nodes: [], Original ATen: []
        torch.ops._C.rms_norm.default(result=buf0, input=buf1, weight=arg3_1, epsilon=1e-06)
        del arg3_1
        buf4 = empty_strided_cuda((s0, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg5_1, buf0, reinterpret_tensor(arg4_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf4)
        del arg4_1
        del arg5_1
        buf5 = reinterpret_tensor(buf0, (s0, 16, 64), (1024, 64, 1), 0); del buf0  # reuse
        buf6 = empty_strided_cuda((s0, 16, 64), (1024, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [cat, cat_2], Original ATen: [aten.cat]
        triton_poi_fused_cat_1_xnumel = 1024*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf4, arg6_1, arg7_1, buf5, buf6, triton_poi_fused_cat_1_xnumel, stream=stream0)
        del arg6_1
        del arg7_1
        buf7 = empty_strided_cuda((s0, 1024), (1024, 1), torch.bfloat16)
    return (buf5, buf6, reinterpret_tensor(buf4, (s0, 16, 64), (3072, 64, 1), 2048), reinterpret_tensor(buf7, (s0, 16, 64), (1024, 64, 1), 0), buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.int32)
    arg1_1 = 2048
    arg2_1 = rand_strided((151936, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg7_1 = rand_strided((32768, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
