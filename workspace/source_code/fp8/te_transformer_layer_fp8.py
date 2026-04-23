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
te_transformer_layer = (
    te.TransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
        self_attn_mask_type="causal",
        layernorm_epsilon=1e-5,
        bias=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    .to(dtype=dtype)
    .cuda()
)
nvtx.range_pop()

recipe = DelayedScaling(
    fp8_format=Format.HYBRID,
    amax_history_len=16,
    amax_compute_algo="max"
)

with te.autocast(enabled=True, recipe=recipe):
    y = te_transformer_layer(x, attention_mask=None)


nvtx.range_push("model execute")
print("TE TransformerLayer + FP8:")
time_te_transformer_layer = utils.speedometer(
    te_transformer_layer,
    x,
    dy,
    forward_kwargs={"attention_mask": None},
    autocast_kwargs={"enabled": True, "recipe": recipe},)
nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop()
