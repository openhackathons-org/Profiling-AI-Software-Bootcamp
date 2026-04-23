# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import Optional
import quickstart_utils as utils
import warnings
from torch.cuda import nvtx
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling


class TEUnfusedMLP(torch.nn.Module):
    """MLP using TE modules."""

    hidden_size: int
    ffn_hidden_size: int

    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.linear1 = te.Linear(hidden_size, ffn_hidden_size, bias=True)
        self.linear2 = te.Linear(ffn_hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.linear2(x)
        return x


class TEUnfusedTransformerLayer(torch.nn.Module):
    """Transformer layer using basic TE modules."""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_eps: float = 1e-5,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = hidden_size // num_attention_heads
        self.ln1 = te.LayerNorm(hidden_size, eps=layernorm_eps)
        self.qkv_projection = te.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attention = utils.DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
        )
        self.projection = te.Linear(hidden_size, hidden_size, bias=True)
        self.dropout1 = torch.nn.Dropout(hidden_dropout)
        self.ln2 = te.LayerNorm(hidden_size, eps=layernorm_eps)
        self.mlp = TEUnfusedMLP(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size)
        self.dropout2 = torch.nn.Dropout(hidden_dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        res = x
        x = self.ln1(x)

        # Fused QKV projection
        qkv = self.qkv_projection(x)
        qkv = qkv.view(qkv.size(0), qkv.size(1), self.num_attention_heads, 3 * self.kv_channels)
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)

        x = self.attention(q, k, v, attention_mask)
        x = self.projection(x)
        x = self.dropout1(x)
        x = res + x

        # Second residual connection
        res = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = self.dropout2(x)

        return x + res

class TEFusedTransformerLayer(torch.nn.Module):
    """Transformer layer using fused TE modules for better performance."""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_eps: float = 1e-5,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = hidden_size // num_attention_heads

        # Fused LayerNorm + QKV projection
        self.ln_qkv = te.LayerNormLinear(hidden_size, 3 * hidden_size, eps=layernorm_eps, bias=True)
        self.attention = te.DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
            attn_mask_type="causal",
        )
        self.projection = te.Linear(hidden_size, hidden_size, bias=True)
        self.dropout1 = torch.nn.Dropout(hidden_dropout)

        # Fused LayerNorm + MLP
        self.ln_mlp = te.LayerNormMLP(hidden_size, ffn_hidden_size, eps=layernorm_eps, bias=True)
        self.dropout2 = torch.nn.Dropout(hidden_dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        res = x

        # Fused LayerNorm + QKV projection
        qkv = self.ln_qkv(x)
        qkv = qkv.view(qkv.size(0), qkv.size(1), self.num_attention_heads, 3 * self.kv_channels)
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)

        x = self.attention(q, k, v, attention_mask)
        x = self.projection(x)
        x = self.dropout1(x)
        x = res + x

        # Fused LayerNorm + MLP
        res = x
        x = self.ln_mlp(x)
        x = self.dropout2(x)

        return x + res



# Layer configuration
hidden_size = 4096
sequence_length = 2048
batch_size = 4
ffn_hidden_size = 16384
num_attention_heads = 32
dtype = torch.float16

torch.cuda.cudart().cudaProfilerStart()
#synthetic data
nvtx.range_push("data to device")
x = torch.rand(sequence_length, batch_size, hidden_size).cuda().to(dtype=dtype)
dy = torch.rand(sequence_length, batch_size, hidden_size).cuda().to(dtype=dtype)
nvtx.range_pop() # Copy to device

#te_unfused = (TEUnfusedTransformerLayer(hidden_size=hidden_size,ffn_hidden_size=ffn_hidden_size, num_attention_heads=num_attention_heads, #).to(dtype=dtype).cuda())

nvtx.range_push("model to device")
#Benchmark TE Fused with FP8:
te_fused_fp8 = (TEFusedTransformerLayer(hidden_size=hidden_size,ffn_hidden_size=ffn_hidden_size,num_attention_heads=num_attention_heads, ).to(dtype=dtype).cuda())
nvtx.range_pop()

recipe = DelayedScaling(
    fp8_format=Format.HYBRID,
    amax_history_len=16,
    amax_compute_algo="max"
)

with te.autocast(enabled=True, recipe=recipe):
    y = te_fused_fp8(x, attention_mask=None)
    #y = te_unfused(x, attention_mask=None)

nvtx.range_push("model to device")
#Benchmark TE Fused with FP8:
#te_fused_fp8 = (TEFusedTransformerLayer(hidden_size=hidden_size,ffn_hidden_size=ffn_hidden_size,num_attention_heads=num_attention_heads, #).to(dtype=dtype).cuda())
#nvtx.range_pop()

nvtx.range_push("model execute")
print("TE Fused + TE Attention + FP8:")
time_te_fused_fp8 = utils.speedometer(
    te_fused_fp8,
    x,
    dy,
    forward_kwargs={"attention_mask": None},
    autocast_kwargs={"enabled": True, "recipe": recipe},)
nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop()