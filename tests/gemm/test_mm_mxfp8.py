import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, mm_mxfp8
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.utils import get_compute_capability


def _run_mm_mxfp8(
    m,
    n,
    k,
    input_dtype,
    is_sf_swizzled_layout,
    out_dtype,
    backend,
    auto_tuning,
    provide_out,
):
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] in [11, 12]:
        pytest.skip("Not tested on SM110/SM120/SM121")
    if compute_capability[0] < 10:
        pytest.skip("mm_mxfp8 is only supported on SM100 and above GPUs.")

    input = torch.randn([m, k], device="cuda", dtype=input_dtype)
    mat2 = torch.randn([n, k], device="cuda", dtype=input_dtype)

    input_mxfp8, input_scale = mxfp8_quantize(input, is_sf_swizzled_layout)
    mat2_mxfp8, mat2_scale = mxfp8_quantize(mat2, is_sf_swizzled_layout)

    reference = torch.mm(input, mat2.T)

    if is_sf_swizzled_layout:
        input_descale = input_scale
        mat2_descale = mat2_scale  # mm_mxfp8 will handle swizzled 1D internally
    else:
        input_descale = input_scale.view(m, k // 32)
        mat2_descale = mat2_scale.view(n, k // 32).t()  # Transpose to (k // 32, n)

    res = torch.empty([m, n], device="cuda", dtype=out_dtype) if provide_out else None

    with autotune(auto_tuning):
        res = mm_mxfp8(
            input_mxfp8,
            mat2_mxfp8.T,  # mm_mxfp8 expects mat2.T (transposed)
            input_descale,
            mat2_descale,
            out=res,
            out_dtype=out_dtype,
            backend=backend,
        )

    assert res.shape == (m, n)
    assert res.dtype == out_dtype
    assert res.device.type == "cuda"
    assert torch.isfinite(res).all(), "Output contains NaN/Inf values"

    min_cos_sim = 0.9
    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    assert cos_sim > min_cos_sim, (
        f"Cosine similarity {cos_sim:.4f} is too low (expected > {min_cos_sim})"
    )


@pytest.mark.parametrize("m", [128, 256, 512, 1024])
@pytest.mark.parametrize("n", [128, 256, 512, 1024])
@pytest.mark.parametrize("k", [128, 256, 512, 1024])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["cutlass"])
@pytest.mark.parametrize("auto_tuning", [True, False])
def test_mm_mxfp8(
    m, n, k, input_dtype, is_sf_swizzled_layout, out_dtype, backend, auto_tuning
):
    _run_mm_mxfp8(
        m,
        n,
        k,
        input_dtype,
        is_sf_swizzled_layout,
        out_dtype,
        backend,
        auto_tuning,
        provide_out=True,
    )


@pytest.mark.parametrize("m", [128, 256])
@pytest.mark.parametrize("n", [128, 256])
@pytest.mark.parametrize("k", [128, 256, 512])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("auto_tuning", [True, False])
def test_mm_mxfp8_backend_auto(
    m, n, k, input_dtype, is_sf_swizzled_layout, out_dtype, auto_tuning
):
    _run_mm_mxfp8(
        m,
        n,
        k,
        input_dtype,
        is_sf_swizzled_layout,
        out_dtype,
        backend="auto",
        auto_tuning=auto_tuning,
        provide_out=False,
    )
