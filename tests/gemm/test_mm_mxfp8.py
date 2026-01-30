import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, mm_mxfp8
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("m", [128, 256, 512])
@pytest.mark.parametrize("n", [128, 256, 512])
@pytest.mark.parametrize("k", [128, 256, 512, 1024])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16])
@pytest.mark.parametrize("backend", ["cudnn"])
@pytest.mark.parametrize("auto_tuning", [True, False])
def test_mm_mxfp8(
    m, n, k, input_dtype, is_sf_swizzled_layout, out_dtype, backend, auto_tuning
):
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] in [11, 12]:
        pytest.skip("Not tested on SM110/SM120/SM121")
    if compute_capability[0] < 10:
        pytest.skip(
            "bmm_mxfp8 with cudnn backend is only supported on SM100 and above GPUs."
        )

    a = torch.randn([m, k], device="cuda", dtype=input_dtype)
    b = torch.randn([k, n], device="cuda", dtype=input_dtype)

    a_mxfp8, a_scale = mxfp8_quantize(a, is_sf_swizzled_layout)
    b_mxfp8, b_scale = mxfp8_quantize(b, is_sf_swizzled_layout)

    # Compute reference result: mm_mxfp8 expects b.t(), so reference should be a @ b.t()
    reference = torch.mm(a, b.t())

    # Prepare scales according to mm_mxfp8's expected format
    # For a_descale: can be 1D swizzled or 2D (m, k//32)
    # For b_descale: mm_mxfp8 expects (k//32, n) format (transposed)
    if is_sf_swizzled_layout:
        # Swizzled: 1D format - mm_mxfp8 will handle reshaping internally
        # For b_descale, we need to handle the transpose format
        # Swizzled b_scale is flattened from (n_padded, k_padded), but mm_mxfp8
        # expects it in a format that can be reshaped to (n_padded, k_padded) then transposed
        a_descale = a_scale
        b_descale = b_scale  # mm_mxfp8 will handle swizzled 1D internally
    else:
        # Non-swizzled: reshape to 2D and transpose b_descale
        a_descale = a_scale.view(m, k // 32)
        b_descale = b_scale.view(n, k // 32).t()  # Transpose to (k // 32, n)

    # Create output tensor
    res = torch.empty([m, n], device="cuda", dtype=out_dtype)

    with autotune(auto_tuning):
        mm_mxfp8(
            a=a_mxfp8,
            b=b_mxfp8.t(),  # mm_mxfp8 expects b.t() (transposed)
            a_descale=a_descale,
            b_descale=b_descale,
            out=res,
            out_dtype=out_dtype,
            backend=backend,
        )

    assert res.shape == (m, n)
    assert res.dtype == out_dtype
    assert res.device == torch.device("cuda")

    min_cos_sim = 0.9
    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    assert cos_sim > min_cos_sim, (
        f"Cosine similarity {cos_sim:.4f} is too low (expected > {min_cos_sim})"
    )
