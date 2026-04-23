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

warnings.filterwarnings("ignore")

class PyTorchMLP(torch.nn.Module):
    """Feed-forward network in Transformer layer.
    Built with plain PyTorch modules.
    """

    hidden_size: int
    ffn_hidden_size: int

    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.linear1 = torch.nn.Linear(hidden_size, ffn_hidden_size, bias=True)
        self.linear2 = torch.nn.Linear(ffn_hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.linear2(x)
        return x

class PyTorchTransformerLayer(torch.nn.Module):
    """Basic Transformer layer using plain PyTorch modules."""

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
        self.ln1 = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps)
        self.qkv_projection = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.attention = utils.DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
        )
        self.projection = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = torch.nn.Dropout(hidden_dropout)
        self.ln2 = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps)
        self.mlp = PyTorchMLP(hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size)

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
        x = self.dropout(x)
        x = res + x

        # Second residual connection
        res = x
        x = self.ln2(x)
        x = self.mlp(x)

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

nvtx.range_push("model to device")
baseline = (
    PyTorchTransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
    )
    .to(dtype=dtype)
    .cuda()
)

print("Baseline PyTorch:")
nvtx.range_push("model execute")
time_baseline = utils.speedometer(baseline, x, dy, forward_kwargs={"attention_mask": None}, )
nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop()
'''
# Layer configuration
hidden_size = 4096
sequence_length = 2048
batch_size = 4
ffn_hidden_size = 16384
num_attention_heads = 32
dtype = torch.float16

#synthetic data
x = torch.rand(sequence_length, batch_size, hidden_size).cuda().to(dtype=dtype)
dy = torch.rand(sequence_length, batch_size, hidden_size).cuda().to(dtype=dtype)

basic_transformer = BasicTransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads)
basic_transformer.to(dtype=dtype).cuda()
print(basic_transformer)

torch.manual_seed(1234)
y = basic_transformer(x, attention_mask=None)

print(utils.speedometer(basic_transformer, x, dy, forward_kwargs={"attention_mask":None}))
'''