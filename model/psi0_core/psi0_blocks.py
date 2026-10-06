"""Psi0 动作专家的基础积木：位置编码 / 时间与条件嵌入 / FiLM / 各类归一化层 / 输出容器。
拆分自 model/psi0.py（原文件备份：已删除的历史单文件备份）。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from diffusers.utils.outputs import BaseOutput
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, CombinedTimestepLabelEmbeddings
from diffusers.models.normalization import FP32LayerNorm

"""本文件由 model/psi0.py 拆分而来（拆分，类体逐字未改）。
历史原单文件已拆分，不随 A 线发布。
"""
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer("pe", torch.clone(pe)) # <-- SONGLIN: this will cause problems when accelerate.prepare
        # self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class CombinedTimestepTextProjEmbeddingsND(CombinedTimestepTextProjEmbeddings):
    """`combined_temb` embedding that also accepts a per-token timestep `(B, Tp)` (RTC).

    diffusers' `time_proj` (get_timestep_embedding) is 1-D only, so a `(B, Tp)` timestep
    would crash the base class. Here we flatten -> embed -> reshape and add the pooled
    (per-sample) instruction projection broadcast over the token axis:

        temb[b, i] = timestep_embed(t[b, i]) + pooled_proj(x[b])

    A 1-D `(B,)` timestep reproduces the base class exactly. Submodule names/params are
    unchanged, so existing combined_temb checkpoints load as-is.
    """

    def forward(self, timestep: torch.Tensor, pooled_projection: torch.Tensor) -> torch.Tensor:
        pooled = self.text_embedder(pooled_projection)  # (B, D)
        if timestep.dim() == 1:                          # (B,) -> (B, D)  [base-class path]
            proj = self.time_proj(timestep)
            temb = self.timestep_embedder(proj.to(dtype=pooled_projection.dtype))
            return temb + pooled
        elif timestep.dim() == 2:                        # (B, Tp) -> (B, Tp, D)  [RTC]
            B, T = timestep.shape
            proj = self.time_proj(timestep.reshape(B * T))
            temb = self.timestep_embedder(proj.to(dtype=pooled_projection.dtype))
            return temb.reshape(B, T, -1) + pooled.unsqueeze(1)
        raise ValueError(f"timestep must be 1-D or 2-D, got shape {tuple(timestep.shape)}")


class FilmConditioning(nn.Module):
    """Layer that adds FiLM conditioning.

    This is intended to be applied after a convolutional layer. It will learn a
    multiplicative and an additive factor to be applied to each channel of the
    convolution's output.

    Conv layer can be rank 2 or 4.

    For further details, see: https://arxiv.org/abs/1709.07871
    """
    
    def __init__(self,
                 in_dim: int,
                 num_channels: int):
        """Constructs a FiLM conditioning layer.

        Args:
            num_channels: Number of filter channels to expect in the input.
        """
        super(FilmConditioning, self).__init__() 
        # Note that we initialize with zeros because empirically we have found
        # this works better than initializing with glorot.
        self._projection_add = nn.Linear(in_dim, num_channels)
        self._projection_mult = nn.Linear(in_dim, num_channels)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, conv_filters, conditioning):
        assert len(conditioning.shape) == 2
        projected_cond_add = self._projection_add(conditioning)
        projected_cond_mult = self._projection_mult(conditioning)

        if len(conv_filters.shape) == 4:
            # [B, D] -> [B, D, 1, 1]
            projected_cond_add = projected_cond_add.unsqueeze(-1).unsqueeze(-1)
            projected_cond_mult = projected_cond_mult.unsqueeze(-1).unsqueeze(-1)
        elif len(conv_filters.shape) == 3:
            # [B, D] -> [B, 1, D]
            projected_cond_add = projected_cond_add.unsqueeze(1)
            projected_cond_mult = projected_cond_mult.unsqueeze(1)
        else:
            assert len(conv_filters.shape) == 2
        
        # Original FiLM paper argues that 1 + gamma centers the initialization at
        # identity transform.
        result = (1 + projected_cond_mult) * conv_filters + projected_cond_add
        return result


class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        # self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))
        self.w = nn.Parameter(w, requires_grad=learnable_w)

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # assert len(x.shape) == 1, "assumes 1d input timestep array"
        # RTC: support 2D timesteps (B,Tp)
        x = x[..., None] * self.w
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        return self.out_net(x)


class ActionTransformerModelOutput(BaseOutput):
    action: "torch.Tensor"  # noqa: F821

    def to_tuple(self):
        return (None, self.action)


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
            assert False, "Not implemented with emb shape == 3, (B, T, D)"
        
        if len(emb.shape) == 2: # (B, D)
            emb = emb.unsqueeze(1) # (B, 1, D)
        elif len(emb.shape) == 3: # (B, T, D)
            pass
        else:
            raise ValueError(f"Invalid shape of emb: {emb.shape}")
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


if torch.__version__ >= "2.1.0":
    LayerNorm = nn.LayerNorm # type:ignore
else:
    # Has optional bias parameter compared to torch layer norm
    # TODO: replace with torch layernorm once min required torch version >= 2.1
    class LayerNorm(nn.Module):
        def __init__(self, dim, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True):
            super().__init__()

            self.eps = eps

            if isinstance(dim, int):
                dim = (dim,)

            self.dim = torch.Size(dim)

            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
            else:
                self.weight = None
                self.bias = None

        def forward(self, input):
            return F.layer_norm(input, self.dim, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, int):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        if len(conditioning_embedding.shape) == 2: # (B, D)
            conditioning_embedding = conditioning_embedding.unsqueeze(1) # (B, 1, D)
        elif len(conditioning_embedding.shape) == 3: # (B, T, D)
            pass
        else:
            raise ValueError(f"Invalid shape of conditioning_embedding: {conditioning_embedding.shape}")
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


@dataclass
class HumanFoundationModelOutput(BaseOutput):
    action: "torch.Tensor"  # noqa: F821
    # Mean per-token L2 norm of the VLM feature fed to the action header (for logging).
    vlm_feat_norm: Optional["torch.Tensor"] = None  # noqa: F821

    def to_tuple(self):
        return (None, self.action)

