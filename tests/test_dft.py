# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import io
from typing import Optional
import pytest

import tensorrt as trt
import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn as nn

from trt_dft_plugins import load_plugins


# This is a simple, but limited, implementation of RFFT/IRFFT custom ops
# which does not require external dependencies. Its purpose is to test TRT export pipeline.
class OnnxRfft2(Function):
    @staticmethod
    def forward(ctx, input: Tensor) -> torch.Value:
        return torch.view_as_real(torch.fft.rfft2(input, dim=(-2, -1), norm="backward"))

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        return g.op(
            "com.microsoft::Rfft", input, normalized_i=0, onesided_i=1, signal_ndim_i=2
        )


class OnnxIrfft2(Function):
    @staticmethod
    def forward(ctx, input: Tensor) -> torch.Value:
        return torch.fft.irfft2(
            torch.view_as_complex(input), dim=(-2, -1), norm="backward"
        )

    @staticmethod
    def symbolic(g: torch.Graph, input: torch.Value) -> torch.Value:
        return g.op(
            "com.microsoft::Irfft", input, normalized_i=0, onesided_i=1, signal_ndim_i=2
        )


class OnnxDft(Function):
    """Standard onnx::DFT (opset >= 20) over the second-to-last axis.

    The last dim carries the real/imag pair (1 = real, 2 = complex), which makes
    the transform axis -2 -- DFT's default -- so no axis/dft_length inputs are
    needed and the node is a single-input node with two int attributes.
    """

    @staticmethod
    def forward(ctx, input: Tensor, inverse: int, onesided: int) -> Tensor:
        if not inverse and onesided:  # RFFT
            return torch.view_as_real(torch.fft.rfft(input[..., 0], dim=-1))
        z = torch.view_as_complex(input.contiguous())
        if not inverse:  # forward C2C
            return torch.view_as_real(torch.fft.fft(z, dim=-1))
        return torch.view_as_real(torch.fft.ifft(z, dim=-1))  # inverse C2C, 1/n

    @staticmethod
    def symbolic(
        g: torch.Graph, input: torch.Value, inverse: int, onesided: int
    ) -> torch.Value:
        return g.op("DFT", input, inverse_i=inverse, onesided_i=onesided)


def dft_rfft2(x: Tensor) -> Tensor:
    """(..., H, W) real -> (..., H, W//2+1, 2), as two standard DFT nodes."""
    t = OnnxDft.apply(x.unsqueeze(-1), 0, 1)
    t = OnnxDft.apply(t.transpose(-3, -2), 0, 0)
    return t.transpose(-3, -2)


def dft_irfft2(t: Tensor) -> Tensor:
    """(..., H, W//2+1, 2) -> (..., H, W) real.

    onesided+inverse is avoided on purpose (onnxruntime rejects it), so the
    Hermitian half is mirrored with standard ops before a full complex inverse.
    """
    t = OnnxDft.apply(t.transpose(-3, -2), 1, 0)
    t = t.transpose(-3, -2)
    mid = t[..., 1:-1, :].flip(-2) * torch.tensor([1.0, -1.0])
    return OnnxDft.apply(torch.cat([t, mid], dim=-2), 1, 0)[..., 0]


@pytest.fixture(scope="session", autouse=True)
def load_trt_plugins():
    load_plugins()


@pytest.fixture()
def trt_logger():
    return trt.Logger(trt.Logger.WARNING)


def export_to_onnx(
    model: nn.Module,
    inp: Tensor,
    verbose: Optional[bool] = True,
    opset_version: int = 15,
) -> bytes:
    with io.BytesIO() as onnx_model:
        # Export to ONNX.
        torch.onnx.export(
            model,
            inp,
            onnx_model,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            opset_version=opset_version,
            verbose=verbose,
        )
        return onnx_model.getvalue()


def build_trt_plan(onnx_model: bytes, logger: trt.ILogger) -> trt.IHostMemory:
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    success = parser.parse(onnx_model)
    assert success, "\n".join(
        str(parser.get_error(i)) for i in range(parser.num_errors)
    )

    config = builder.create_builder_config()
    return builder.build_serialized_network(network, config)


def run_trt_inference(
    trt_plan: trt.IHostMemory, x: Tensor, y: Tensor, logger: trt.ILogger
) -> Tensor:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(trt_plan)
    context = engine.create_execution_context()
    x = x.cuda()
    y_device = y.device
    y = y.cuda()
    buffers = [x.data_ptr(), y.data_ptr()]
    context.execute_v2(buffers)
    return y.to(y_device)


def test_plugins_load():
    loaded_plugins = {p.name for p in trt.get_plugin_registry().plugin_creator_list}
    assert "Rfft" in loaded_plugins
    assert "Irfft" in loaded_plugins
    assert "DFT" in loaded_plugins


# (inverse, onesided): RFFT, forward C2C, inverse C2C. onesided+inverse is not
# supported -- see dft_irfft2.
@pytest.mark.parametrize("inverse,onesided", [(0, 1), (0, 0), (1, 0)])
@pytest.mark.parametrize("n", [4, 8])
@pytest.mark.parametrize("batch_dims", [(1, 3), (2, 1)])
def test_dft(trt_logger, inverse, onesided, n, batch_dims):
    class DftModel(nn.Module):
        def forward(self, x):
            return OnnxDft.apply(x, inverse, onesided)

    torch.manual_seed(1)
    # real input (trailing 1) for RFFT, complex (trailing 2) otherwise
    x = torch.randn(*batch_dims, n, 1 if onesided else 2)

    onnx_model = export_to_onnx(DftModel(), x, opset_version=20)
    trt_plan = build_trt_plan(onnx_model, trt_logger)

    y_expected = OnnxDft.apply(x, inverse, onesided)
    y = run_trt_inference(trt_plan, x, torch.empty_like(y_expected), trt_logger)

    assert torch.allclose(y_expected, y, atol=1e-5)


@pytest.mark.parametrize("h,w", [(4, 8), (8, 8)])
def test_dft_rfft2_roundtrip(trt_logger, h, w):
    """The shape the exporter actually emits: rfft2 -> irfft2 must reconstruct."""

    class Roundtrip(nn.Module):
        def forward(self, x):
            return dft_irfft2(dft_rfft2(x))

    torch.manual_seed(1)
    x = torch.randn(2, 3, h, w)

    onnx_model = export_to_onnx(Roundtrip(), x, opset_version=20)
    trt_plan = build_trt_plan(onnx_model, trt_logger)

    y = run_trt_inference(trt_plan, x, torch.empty_like(x), trt_logger)

    assert torch.allclose(x, y, atol=1e-4)
    # ... and match torch.fft, which is what the model computed before the
    # switch to standard DFT nodes.
    want = torch.fft.irfft2(
        torch.fft.rfft2(x, dim=(-2, -1), norm="backward"), dim=(-2, -1), norm="backward"
    )
    assert torch.allclose(want, y, atol=1e-4)


@pytest.mark.parametrize("dft_dim1", [1, 2])
@pytest.mark.parametrize("dft_dim2", [4])
@pytest.mark.parametrize("num_c", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_rfft2(trt_logger, dft_dim1, dft_dim2, num_c, batch_size):
    class RfftModel(nn.Module):
        def forward(self, x):
            return OnnxRfft2.apply(x)

    model = RfftModel()

    torch.manual_seed(1)
    x = torch.randn(batch_size, num_c, dft_dim1, dft_dim2)

    # 1. Export to ONNX.
    onnx_model = export_to_onnx(model, x)

    # 2. Build TRT plan from ONNX.
    trt_plan = build_trt_plan(onnx_model, trt_logger)

    # 3. Run TRT inference.
    y_expected = OnnxRfft2.apply(x)
    # y stores the output of the RFFT2 ops.
    y = torch.empty_like(y_expected)
    y = run_trt_inference(trt_plan, x, y, trt_logger)

    # Both implementations should produce the same result.
    assert torch.allclose(y_expected, y)


@pytest.mark.parametrize("model_cls", [OnnxRfft2, OnnxIrfft2])
def test_dynamic_batch(trt_logger, model_cls):
    class Model(nn.Module):
        def forward(self, x):
            return model_cls.apply(x)

    torch.manual_seed(1)
    x = torch.randn(2, 3, 2, 4)
    if model_cls is OnnxIrfft2:
        x = OnnxRfft2.apply(x)

    with io.BytesIO() as f:
        torch.onnx.export(
            Model(),
            x,
            f,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            opset_version=15,
            input_names=["x"],
            dynamic_axes={"x": {0: "batch"}},
        )
        onnx_model = f.getvalue()

    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    assert parser.parse(onnx_model)

    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    shape = tuple(x.shape[1:])
    profile.set_shape("x", (1, *shape), (2, *shape), (4, *shape))
    config.add_optimization_profile(profile)
    trt_plan = builder.build_serialized_network(network, config)

    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(trt_plan)
    context = engine.create_execution_context()
    for batch_size in (1, 3, 4):
        xb = torch.randn(batch_size, *shape)
        y_expected = model_cls.apply(xb)
        xb_gpu = xb.cuda()
        y = torch.empty_like(y_expected).cuda()
        context.set_binding_shape(0, tuple(xb.shape))
        context.execute_v2([xb_gpu.data_ptr(), y.data_ptr()])
        assert torch.allclose(y_expected, y.cpu())


@pytest.mark.parametrize("dft_dim1", [1, 2])
@pytest.mark.parametrize("dft_dim2", [4])
@pytest.mark.parametrize("num_c", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_irfft2(trt_logger, dft_dim1, dft_dim2, num_c, batch_size):
    class IrfftModel(nn.Module):
        def forward(self, x):
            return OnnxIrfft2.apply(x)

    model = IrfftModel()

    torch.manual_seed(1)
    x = torch.randn(batch_size, num_c, dft_dim1, dft_dim2)

    # Compute RFFT first.
    y = OnnxRfft2.apply(x)

    # 1. Export to ONNX.
    onnx_model = export_to_onnx(model, y)

    # 2. Build TRT plan from ONNX.
    trt_plan = build_trt_plan(onnx_model, trt_logger)

    # 3. Run TRT inference.
    x_expected = OnnxIrfft2.apply(y)
    # x_actual stores the output of the IRFFT2 ops.
    x_actual = torch.empty_like(x_expected)
    x_actual = run_trt_inference(trt_plan, y, x_actual, trt_logger)

    # Both implementations should produce the same result.
    assert torch.allclose(x_expected, x_actual)
