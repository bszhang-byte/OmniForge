"""Psi0 动作专家的联合注意力：MM-DiT 风格 processor 与 transformer 块。
动作流（to_q/k/v）与观测条件流（add_q/k/v_proj）拼接做双向注意力后拆分。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward
from diffusers.utils import logging
logger = logging.get_logger(__name__)

from model.psi0_core.psi0_blocks import AdaLayerNormZero, AdaLayerNormContinuous

"""本文件由 model/psi0.py 拆分而来（拆分，类体逐字未改）。
历史原单文件已拆分，不随 A 线发布。
"""
class JointVLAAttnProcessor: #(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        context_pre_only=None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("JointAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        # self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        # self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        # self.out_dim = out_dim if out_dim is not None else query_dim

        # self.to_q_lang = nn.Linear(query_dim, self.inner_dim, bias=bias)
        # self.to_k_lang = nn.Linear(query_dim, self.inner_kv_dim, bias=bias)
        # self.to_v_lang = nn.Linear(query_dim, self.inner_kv_dim, bias=bias)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)

        assert attn.to_k is not None
        key = attn.to_k(hidden_states)

        assert attn.to_v is not None
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
            if attention_mask is not None:
                assert attention_mask.dtype == torch.float32
                attn_mask = torch.cat([
                    torch.ones(hidden_states.shape[0], 1, 1, hidden_states.shape[1], device=attention_mask.device).to(torch.bool),  # (B, 1, 1, S1)
                    (attention_mask == 1)[:, None, None, :]  # (B, 1, 1, S2)
                ], dim=-1)
            else:
                attn_mask = None

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attn_mask) # type: ignore
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim) # type: ignore
        hidden_states = hidden_states.to(query.dtype) # type: ignore
 
        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        ) # type: ignore
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states, encoder_hidden_states
    

class VLATransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
        training_phase: str = "joint",  # "joint", "action", "traj2d", "action_mini"
    ):
        super().__init__()

        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        self.norm1_act = AdaLayerNormZero(dim)
        if context_norm_type == "ada_norm_continous":
            """ self.norm1_lang = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm") """
            self.norm1_obs = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm")
        elif context_norm_type == "ada_norm_zero":
            """ self.norm1_lang = AdaLayerNormZero(dim) """
            self.norm1_obs = AdaLayerNormZero(dim)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )
        
        if hasattr(F, "scaled_dot_product_attention"):
            processor = JointVLAAttnProcessor(
                query_dim=dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                context_pre_only=context_pre_only,
            )
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor, # type: ignore
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.norm2_act = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_act = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            """ self.norm2_lang = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_lang = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate") """

            self.norm2_obs = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_obs = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            """ self.norm2_lang = None
            self.ff_lang = None """

            self.norm2_obs= None
            self.ff_obs = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self, 
        # hidden_states: torch.FloatTensor,  #  V
        # lang_hidden_states: torch.FloatTensor,  # L
        action_hidden_states: torch.Tensor, # A
        obs_hidden_states: torch.Tensor, # O
        temb: torch.Tensor,
        obs_token_mask: Optional[torch.Tensor] = None,
        # obs_pos_emb: Optional[torch.FloatTensor] = None,
        # time_enc: Optional[torch.FloatTensor] = None,
        joint_attention_kwargs=None,
    ):
        # obs_hidden_states = self.encoder_module(src=obs_hidden_states, pos=obs_pos_emb)
        # action_hidden_states = self.decoder_module(x=action_hidden_states, t=time_enc, cond=obs_hidden_states) # recursive: _TransformerDecoder.forward
        
        norm_action_hidden_states, gate_msa_act, shift_mlp_act, scale_mlp_act, gate_mlp_act = self.norm1_act(action_hidden_states, emb=temb)

        if self.context_pre_only:
            """ norm_lang_hidden_states = self.norm1_lang(lang_hidden_states, temb) """
            norm_obs_hidden_states = self.norm1_obs(obs_hidden_states, temb[:,-1] if len(temb.shape) > 2 else temb)

            gate_msa_obs, shift_mlp_obs, scale_mlp_obs, gate_mlp_obs = None, None, None, None
        else:
            """ norm_lang_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_lang(
                lang_hidden_states, emb=temb
            ) """
            norm_obs_hidden_states, gate_msa_obs, shift_mlp_obs, scale_mlp_obs, gate_mlp_obs = self.norm1_obs(obs_hidden_states, emb=temb[:,-1] if len(temb.shape) > 2 else temb)

        act_attn_output, obs_attn_output = self.attn( ## indirectly calls JointAttnProcessor2_0.__call__
            # hidden_states=None,
            # lang_hidden_states=norm_lang_hidden_states,
            hidden_states=norm_action_hidden_states,
            encoder_hidden_states=norm_obs_hidden_states,
            attention_mask=obs_token_mask
            # **joint_attention_kwargs,
        )

        # Action feed forward
        act_attn_output = gate_msa_act * act_attn_output
        action_hidden_states = action_hidden_states + act_attn_output

        norm_action_hidden_states = self.norm2_act(action_hidden_states)
        norm_action_hidden_states = norm_action_hidden_states * (1 + scale_mlp_act) + shift_mlp_act
        ff_action_output = self.ff_act(norm_action_hidden_states)

        ff_action_output = gate_mlp_act * ff_action_output
        action_hidden_states = action_hidden_states + ff_action_output

        # attention outputs for the `obs_hidden_states`.
        if self.context_pre_only:
            # encoder_hidden_states = None
            obs_hidden_states = None # type:ignore
        else:
            assert gate_msa_obs is not None and \
                shift_mlp_obs is not None and \
                    scale_mlp_obs is not None and \
                        gate_mlp_obs is not None
            """ # lang forward
            lang_attn_output = c_gate_msa.unsqueeze(1) * lang_attn_output
            lang_hidden_states = lang_hidden_states + lang_attn_output
            norm_lang_hidden_states = self.norm2_lang(lang_hidden_states)
            norm_lang_hidden_states = norm_lang_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_lang, norm_lang_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_lang(norm_lang_hidden_states)
            lang_hidden_states = lang_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output """
            
            # obs feedforward
            assert self.norm2_obs is not None

            obs_attn_output = gate_msa_obs * obs_attn_output
            obs_hidden_states = obs_hidden_states + obs_attn_output

            norm_obs_hidden_states = self.norm2_obs(obs_hidden_states)
            norm_obs_hidden_states = norm_obs_hidden_states * (1 + scale_mlp_obs) + shift_mlp_obs

            assert self.ff_obs is not None
            ff_obs_output = self.ff_obs(norm_obs_hidden_states)
            ff_obs_output = gate_mlp_obs * ff_obs_output
            obs_hidden_states = obs_hidden_states + ff_obs_output

        return action_hidden_states, obs_hidden_states
    
