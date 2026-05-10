"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

from flashinfer import ActivationType, RoutingMethodType, fp4_quantize
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
from flashinfer.utils import device_support_pdl, get_compute_capability

NVFP4_GLOBAL_SCALE = torch.tensor([448.0 * 6.0])


def _quantize_fp4(t: torch.Tensor, *, batched: bool):
    """NVFP4-quantize a 2D (token,hidden) or 3D (expert,M,K) tensor."""
    quantized, scale = fp4_quantize(
        t,
        NVFP4_GLOBAL_SCALE.to(t.device),
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=batched,
    )
    if batched:
        scale = scale.view(torch.float8_e4m3fn).reshape(t.shape[0], t.shape[1], -1)
    else:
        scale = scale.view(torch.float8_e4m3fn).reshape(t.shape[0], -1)
    return quantized, scale


def _make_fp4_moe_inputs(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    device: torch.device,
    seed: int,
):
    """Build NVFP4-quantized hidden states + weights + per-expert scales."""
    torch.manual_seed(seed)
    hs_bf16 = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    )
    hs_quant, hs_scale = _quantize_fp4(hs_bf16, batched=False)

    w13_bf16 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    w13, w13_scale = _quantize_fp4(w13_bf16, batched=True)

    w2_bf16 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    w2, w2_scale = _quantize_fp4(w2_bf16, batched=True)

    g = 1.0 / 448.0 / 6.0
    out_scale = torch.full((num_experts,), g * g, device=device)
    return {
        "hs": hs_quant,
        "hs_scale": hs_scale,
        "w13": w13,
        "w13_scale": w13_scale,
        "w2": w2,
        "w2_scale": w2_scale,
        "out_scale": out_scale,
    }


def _call_fp4_moe(
    *,
    routing_logits,
    fx,
    num_experts,
    top_k,
    intermediate_size,
    local_expert_offset,
    local_num_experts,
    enable_pdl,
    output=None,
):
    return trtllm_fp4_block_scale_moe(
        routing_logits,
        None,  # routing_logits, routing_bias
        fx["hs"],
        fx["hs_scale"],
        fx["w13"],
        fx["w13_scale"],
        None,
        None,
        None,
        None,  # gemm1_bias, alpha, beta, clamp_limit
        fx["w2"],
        fx["w2_scale"],
        None,  # gemm2_bias
        fx["out_scale"],
        fx["out_scale"],
        fx["out_scale"],
        num_experts,
        top_k,
        None,
        None,  # top_k, n_group, topk_group
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        None,  # routed_scaling_factor
        RoutingMethodType.Renormalize.value,
        True,  # do_finalize
        enable_pdl,
        ActivationType.Swiglu.value,
        None,  # per_token_scale
        output,
    )[0]


@pytest.mark.parametrize(
    "num_experts,ep_size",
    [
        pytest.param(8, 1, id="ep1_e8"),  # baseline
        pytest.param(8, 2, id="ep2_e8"),
        pytest.param(64, 4, id="ep4_e64"),
    ],
)
def test_fp4_moe_ep_partial_sum_equivalence(num_experts: int, ep_size: int):
    """Sum of per-rank EP=K outputs must equal the EP=1 reference (top_k=1)."""
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] != 10:
        pytest.skip("trtllm-gen NVFP4 MoE requires SM100/SM103 (Blackwell).")
    assert num_experts % ep_size == 0

    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)
    num_tokens, hidden_size, intermediate_size, top_k = 32, 1024, 1024, 1

    fx = _make_fp4_moe_inputs(
        num_tokens, hidden_size, intermediate_size, num_experts, device, seed=42
    )
    g = torch.Generator(device="cpu").manual_seed(43)
    selected = torch.randint(0, num_experts, (num_tokens,), generator=g).to(device)
    logits = torch.full((num_tokens, num_experts), -10.0, device=device)
    logits[torch.arange(num_tokens, device=device), selected] = 10.0
    routing_logits = logits.to(torch.bfloat16)

    reference = _call_fp4_moe(
        routing_logits=routing_logits,
        fx=fx,
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        enable_pdl=enable_pdl,
    ).to(torch.float)
    assert not torch.isnan(reference).any() and not torch.isinf(reference).any()
    if ep_size == 1:
        return

    local_num_experts = num_experts // ep_size
    accumulated = torch.zeros_like(reference)
    for rank in range(ep_size):
        rank_out = torch.zeros(
            (num_tokens, hidden_size), device=device, dtype=torch.bfloat16
        )
        _call_fp4_moe(
            routing_logits=routing_logits,
            fx=fx,
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=intermediate_size,
            local_expert_offset=rank * local_num_experts,
            local_num_experts=local_num_experts,
            enable_pdl=enable_pdl,
            output=rank_out,
        )
        rank_f = rank_out.to(torch.float)
        assert not torch.isnan(rank_f).any() and not torch.isinf(rank_f).any()
        accumulated += rank_f

    mask = torch.isclose(accumulated, reference, rtol=1e-2, atol=1e-2)
    mismatch_pct = (~mask).float().mean().item() * 100.0
    assert mismatch_pct < 5.0, (
        f"EP={ep_size} (e={num_experts}) sum diverged from EP=1 reference: "
        f"{mismatch_pct:.2f}% mismatch, "
        f"max_abs_diff={(accumulated - reference).abs().max().item():.4f}"
    )
