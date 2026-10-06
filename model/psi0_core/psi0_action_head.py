"""Psi0 动作专家（ActionTransformerModel）与输入输出投影。
流匹配去噪头：观测投影（VLM 隐状态+本体状态 → 上下文 token）
+ 动作块投影（noisy action → 动作 token）→ N 层联合注意力块 → adaLN 输出层。
"""
import os
import copy
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Dict, Any
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward
from diffusers.utils import logging
logger = logging.get_logger(__name__)

from model.psi0_core.psi0_blocks import (
    _PositionalEncoding, _TimeNetwork, FilmConditioning,
    CombinedTimestepTextProjEmbeddingsND, ActionTransformerModelOutput,
)
from model.psi0_core.psi0_attention import VLATransformerBlock

"""本文件由 model/psi0.py 拆分而来（拆分，类体逐字未改）。
历史原单文件已拆分，不随 A 线发布。
"""
class ObservationProjection(nn.Module): # FIXME naming
    """ VLT observation projector

    Args:
        ...
    """

    def __init__(
        self,
        action_pred_horizon: int = 50, # Tp
        action_dim: int = 7, # in dim
        output_dim: int = 512, # out dim
        hidden_dim: int = 1280,
        n_conditions: int = 1, # num image of contitions
        timestep_in_dim: int = 320,
        timestep_flip_sin_to_cos: bool = True,
        timestep_freq_shift: int = 0,
        token_fusion: str = "concat",
        resnet_store_path: str = "cache/visual_features/resnet18/IN_1M_resnet18.pth",
        odim: int = 32,
        view_feature_dim: int = 1920,
        use_film: bool = False
    ):
        super().__init__()

        # TODO 1. compare
        
        # self.time_net = _TimeNetwork(time_dim=256, out_dim=512)
        
        # self.time_proj = Timesteps(timestep_in_dim, timestep_flip_sin_to_cos, timestep_freq_shift)
        # self.time_embedding = TimestepEmbedding(timestep_in_dim, output_dim, act_fn="silu")
        self.action_pred_horizon = action_pred_horizon
        self.action_dim = action_dim
        # self.ac_proj = nn.Sequential(
        #     nn.Linear(action_dim, action_dim),
        #     nn.GELU(approximate="tanh"),
        #     nn.Linear(action_dim, output_dim),
        # )

        # _DiTNoiseNet.__init__
        self.enc_pos = _PositionalEncoding(d_model=output_dim)

        

        # agent.__init__
        n_cams = 0 # 1
        
        imgs_per_cam = 1
        self.camera_indices = [] # [0]
        self._n_cams = n_cams
        self._share_cam_features = False # model_cfg.share_cam_features
        early_fusion = False
        text_dim = 768
        # odim = 15
        self.token_fusion = token_fusion

        features = None
        feat_list = [copy.deepcopy(features) for _ in range(0, n_conditions)] # + [copy.deepcopy(features) for _ in range(1, n_cams)] + [copy.deepcopy(features) for _ in range(0, n_conditions)]
        self.visual_features = nn.ModuleList(feat_list)
        self.views_proj = nn.Linear(view_feature_dim, output_dim, bias=True) # TODO configure 1920 # boqian_fix
        
        self.early_fusion = early_fusion
        self.imgs_per_cam = imgs_per_cam
        imgs_per_cam = 1 if early_fusion else imgs_per_cam
        self._n_conditions = n_conditions

        self.embed_proj = nn.Identity()
        if self.token_fusion != "perceiver" and self._n_conditions > 0:
            assert features is not None
            self._token_dim = features.embed_dim
            if output_dim != self._token_dim:
                self.embed_proj = nn.Linear(self._token_dim, output_dim, bias=False)
        else:
            self._token_dim = 0

        self._n_tokens = imgs_per_cam * n_cams * self._token_dim + n_conditions * self._token_dim
        self.traj_n_token = self._token_dim
        
        self.feat_norm = nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-6) # feat_norm = None
        self.feat_traj_norm = nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-6) # feat_norm = None
        # self.traj2d_proj = nn.ModuleList([
        #     nn.Linear(output_dim, output_dim) for _ in range(2)
        # ])
        # self.traj2d_proj = nn.Conv2d(
        #     16, output_dim, kernel_size=(2, 2), stride=2, bias=True
        # )

        if self.token_fusion == "cross":
            
            self.attn = Attention(
                query_dim=self._token_dim,
                cross_attention_dim=1920,
                dim_head=64,
                out_dim=output_dim,
                heads=output_dim//64,
                bias=True,
                # pre_only=True,
                eps=1e-6
            )
        elif self.token_fusion == "perceiver":
            num_queries = 256
            self.latents1 = nn.Parameter(torch.randn(1, num_queries, hidden_dim) / hidden_dim**0.5)
            # self.proj_in1 = nn.Linear(1920, hidden_dim)  -> views_proj (see above)
            self.cross_attn1 = Attention(
                query_dim=hidden_dim,
                cross_attention_dim=hidden_dim,
                dim_head=64,
                out_dim=output_dim,
                heads=output_dim//64,
                bias=True,
                # pre_only=True,
                eps=1e-6
            )
            self.latents2 = nn.Parameter(torch.randn(1, num_queries, hidden_dim) / hidden_dim**0.5)
            self.proj_in2 = nn.Linear(self._token_dim, hidden_dim)
            self.cross_attn2 = Attention(
                query_dim=hidden_dim,
                cross_attention_dim=hidden_dim,
                dim_head=64,
                out_dim=output_dim,
                heads=output_dim//64,
                bias=True,
                # pre_only=True,
                eps=1e-6
            )

        self.film = FilmConditioning(2048, output_dim) if use_film else None

        self._obs_strat = "add_token"
        self._n_tokens += 1

        self._obs_proc = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(odim, output_dim)
        )
        dropout = 0.1 # @see model_cfg.dropout
        linear_proj = nn.Identity() # build (optional) token feature projection layer 
        norm = nn.Identity() # feat_norm = None
        self.post_proc = nn.Sequential(linear_proj, norm, nn.Dropout(dropout))

    def forward(self, 
                # noisy_actions: torch.Tensor, 
                # timestep: torch.Tensor, 
                # temp
                views, 
                obs, 
                traj2ds=None, 
                # ac_flat = None, 
                # mask_flat = None, 
                text_embeddings = None, 
                vlm_attn_mask=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            noisy_actions (`torch.Tensor`):
                Image features.
            timestep (`torch.Tensor`):
                Timestep in denoising process.
        Returns:
            `Tuple`[`torch.Tensor`, `torch.Tensor`]: The pair (latents, timestep_emb).
        """
        # DiffusionTransformerAgent.forward
        vlm_token_len = views.shape[1] * views.shape[2]  # V*S VLM tokens, right-padded
        s_t, out_attn_mask = self.tokenize_obs(views, obs, traj2ds, text_embeddings=text_embeddings, vlm_attn_mask=vlm_attn_mask) # (B, S, d_model)
        # _DiTNoiseNet.forward_enc
        s_t = s_t.transpose(0, 1) # (S, B, d_model)
        s_t = self._add_obs_pos(s_t, vlm_token_len) # (S, B, d_model)
        return s_t.transpose(0,1), out_attn_mask # (B,S,d_model)

    def _add_obs_pos(self, s_t, vlm_token_len):
        """Positional encoding for the [VLM tokens ..., obs token(s)] sequence.
            Only the observation tokens receive positional encodings. 
            The VLM tokens retain their original positions as determined by the VLM's own positional encoding.
        """
        n_obs = s_t.shape[0] - vlm_token_len
        if n_obs <= 0:
            return s_t
        obs_pos = self.enc_pos.pe[:n_obs].to(s_t.dtype)  # (n_obs, 1, d_model)
        return torch.cat([s_t[:vlm_token_len], s_t[vlm_token_len:] + obs_pos], dim=0)

    def tokenize_obs(self, views, obs, traj2ds = None, flatten=False, text_embeddings=None, vlm_attn_mask=None):
        
        view_tokens = self.views_proj(views)
        B,V,S,D = view_tokens.shape
        
        if traj2ds is not None:
            traj_tokens = self.embed({}, {f"cond{i}": traj2ds[:, i] for i in range(traj2ds.shape[1])})

            if self.token_fusion == "concat":
                traj_tokens = self.embed_proj(traj_tokens) #(B,S,d_model)
                tokens = torch.cat([view_tokens.view(B,V*S,D), traj_tokens], dim=1) # (B, tok1+tok2, 1536)
            elif self.token_fusion == "cross":  
                traj_tokens_list, view_tokens_list = [], []
                for v_idx in range(traj2ds.shape[1]):
                    # traj2d --> cross attend --> view features
                    view_tokens = self.attn(
                        hidden_states=traj_tokens[:, v_idx*self.traj_n_token, (v_idx+1)*self.traj_n_token],
                        encoder_hidden_states=views[:, v_idx],
                    )
                    traj_tokens_list.append(traj_tokens)
                    view_tokens_list.append(view_tokens)

                traj_tokens = torch.cat(traj_tokens_list, dim=1)
                traj_tokens = self.embed_proj(traj_tokens)
                view_tokens = torch.cat(view_tokens_list, dim=1)
                tokens = torch.cat([view_tokens, traj_tokens], dim=1) # (B, tok1+tok2, 1536)
            elif self.token_fusion == "perceiver":
                latents1 = self.latents1.repeat(B, 1, 1)
                # views = self.proj_in1(views)
                # views = views + text_embeddings[:, None]
                tokens1 = self.cross_attn1(
                    hidden_states=latents1,
                    encoder_hidden_states=view_tokens.view(B, V*S, D),
                )
                latents2 = self.latents2.repeat(B, 1, 1)
                traj_tokens = self.proj_in2(traj_tokens)
                # traj2ds = traj2ds + text_embeddings[:, None]
                tokens2 = self.cross_attn2(
                    hidden_states=latents2,
                    encoder_hidden_states=traj_tokens,
                )
                tokens = torch.cat([tokens1, tokens2], dim=1) # (B, tok1+tok2, 1536)
            else:
                raise ValueError
        else:
            assert self._n_conditions == 0, "inconsistent confg"
            tokens = view_tokens.view(B,V*S,D)

        if (self.film is not None) and (text_embeddings is not None):
            tokens = self.film(tokens, text_embeddings)

        if self._obs_strat == "add_token":
            obs_token = self._obs_proc(obs)#[:, None]
            tokens = torch.cat((tokens, obs_token), 1)
            obs_token_len = obs_token.shape[1]
            vlm_attn_mask = (
                torch.cat([vlm_attn_mask, torch.ones((B, obs_token_len), device=vlm_attn_mask.device)], 1) 
                if vlm_attn_mask is not None else None
            )
        elif self._obs_strat == "pad_img_tokens":
            obs = self._obs_proc(obs)
            obs = obs[:, None].repeat((1, tokens.shape[1], 1))
            tokens = torch.cat((obs, tokens), 2)
            vlm_attn_mask = None
        else:
            assert self._obs_strat is None

        tokens = self.post_proc(tokens)
        if flatten:
            return tokens.reshape((tokens.shape[0], -1)), vlm_attn_mask
        return tokens, vlm_attn_mask
    
    def embed(self, imgs, conditions=None):
        """ SONGLIN: DOWNSCALE == 2**5 = 32, total 5 conv2d with stride(2) """
        def embed_helper(net, im):
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16): # TODO check if this is needed
            if self.early_fusion and len(im.shape) == 5:
                T = im.shape[1]
                im = torch.cat([im[:, t] for t in range(T)], 1)
                embeds = net(im)
            elif len(im.shape) == 5:
                B, T, C, H, W = im.shape
                embeds = net(im.reshape((B * T, C, H, W)))
                embeds = embeds.reshape((B, -1, net.embed_dim))
            else:
                assert len(im.shape) == 4
                embeds = net(im)
            return embeds

        if self._share_cam_features:
            embeds = [
                embed_helper(self.visual_features, imgs[f"cam{i}"])
                for i in self.camera_indices
            ]
            if conditions is not None:
                assert self._n_conditions > 0
                embeds += [
                    embed_helper(self.visual_features, conditions[f"cond{i}"]) 
                    for i in range(self._n_conditions)
                ]
        else:
            assert conditions is not None
            # img_encoders = self.visual_features[:self._n_cams] # sum([v.sum().item() for k, v in img_encoders.named_parameters()])
            # cond_encoders = self.visual_features[self._n_cams:]
            cond_encoders = self.visual_features
            # [
            #     embed_helper(net, imgs[f"cam{i}"])
            #     for i, net in zip(self.camera_indices, img_encoders)
            # ] +
            embeds = [
                embed_helper(net, conditions[f"cond{0}"])
                for i, net in enumerate(cond_encoders)
            ]

        """ if self.training: # FIXME better handle this thing
            return torch.cat(embeds, dim=1) """
        assert conditions is not None
        in_dtype = next(iter(imgs.values())).dtype if len(imgs) > 0 else next(iter(conditions.values())).dtype
        embeds = torch.cat(embeds, dim=1).to(dtype=in_dtype) # (B, S, d_model)
        return embeds


class ActionProjectionIn(nn.Module):
    def __init__(
        self,
        action_pred_horizon,
        action_dim,
        output_dim,
    ):  
        super().__init__()
        self.action_pred_horizon = action_pred_horizon
        self.action_dim = action_dim

        self.ac_proj = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_dim, output_dim),
        )
        self.dec_pos = nn.Parameter(torch.empty(action_pred_horizon, output_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.dec_pos.data)

    def forward(self, noisy_actions: torch.Tensor, 
        # timestep
        ) -> torch.Tensor:
        B = noisy_actions.shape[0]
        # action_dim = self._dit_obs_proj.action_dim
        # time_enc = self.time_net(timestep)
        noise_acs = noisy_actions.reshape(B, -1, self.action_dim) # DiffusionTransformerAgent.forward
        ac_tokens = self.ac_proj(noise_acs) # noisy_actions, _DiTNoiseNet.forward
        # ac_tokens = ac_tokens.transpose(0, 1) # (B, L, d_model) _DiTNoiseNet.forward_dec
        action_hidden_states = ac_tokens + self.dec_pos.unsqueeze(0)
        return action_hidden_states#, time_enc


class ActionProjectionOut(nn.Module): # _FinalLayer
    def __init__(self, hidden_size, action_dim, final_layer_norm: bool = True):
        super().__init__()
        self.final_layer_norm = final_layer_norm
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        if final_layer_norm:
            # DiT convention: zero-init the modulation so the head starts as a
            # plain LayerNorm (scale=shift=0 -> x*(1+0)+0) instead of a random
            # unbounded gate.
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)
        # self.reset_parameters()

    def forward(self, x, t, cond=None):
        # process the conditioning vector first
        # cond = torch.mean(cond, dim=1)
        # cond = cond + t

        if len(t.shape) == 2: # (B, D)
            t = t.unsqueeze(1) # (B, 1, D)
        elif len(t.shape) == 3: # (B, T, D)
            pass
        else:
            raise ValueError(f"Invalid shape of t: {t.shape}")
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=-1) # cond -> t
        if self.final_layer_norm:
            x = self.norm_final(x) * (1 + scale) + shift
        else:
            # Legacy head: the unnormalized residual stream times an unbounded
            # gate, with no identity path. d(loss)/dx scales with `scale` and
            # vice versa, so the two grow into each other -- this is what could drive
            # the grad-norm blowups.
            x = x * scale + shift
        x = self.linear(x)
        # return x.transpose(0, 1)
        return x


class ActionTransformerModel(
    ModelMixin, ConfigMixin #, VLTTransformerActionMixin
    # PeftAdapterMixin, FromOriginalModelMixin, ReprodMixin, SD3Transformer2DLoadersMixin, 
):  
    """  An action transformer model """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["ActionTransformerBlock"]
    # _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @classmethod
    def from_pretrained(cls,  pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        # model = cls.from_config(config=None, **kwargs)
        model = ActionTransformerModel()
        torch_dtype = kwargs.pop("torch_dtype", torch.float32)
        if pretrained_model_name_or_path is not None and "checkpoints" in str(pretrained_model_name_or_path):
            if os.path.exists(pretrained_model_name_or_path):
                 # NOTE use accelerator.checkpointing 
                from safetensors.torch import load_file
                file_path = f"{pretrained_model_name_or_path}/model.safetensors"
                assert os.path.exists(file_path), f"Checkpoint {file_path} does not exist."
                model_state_dict = load_file(file_path)
                missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
                
                """ # load all checkpoint data, including weights, optimizer, etc.
                load_dict = torch.load(pretrained_model_name_or_path, weights_only=False) 
                from torch.nn.parallel import DistributedDataParallel as DDP
                model = model.module if isinstance(model, DDP) else model
                missing_keys, unexpected_keys = model.load_state_dict(load_dict["model"], strict=False)  """

                if len(unexpected_keys) > 0:
                    logger.info(f"Unexpected keys in the checkpoint: {unexpected_keys}. ")
                if len(missing_keys) > 0:
                    logger.info(f"Missing keys in the checkpoint: {missing_keys}.")

                logger.info(f"Loaded model from {pretrained_model_name_or_path}")
            else:
                raise ValueError(f"Checkpoint {pretrained_model_name_or_path} does not exist. ")
        
        model.to(torch_dtype)
        return model

    @register_to_config
    def __init__(
        self,
        # sample_size: int = 128,
        # patch_size: int = 2,
        # in_channels: int = 16,
        # num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        # joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1536,
        pooled_projection_dim: int = 2048,
        # out_channels: int = 16,
        pos_embed_max_size: int = 192,
        dual_attention_layers: Tuple[
            int, ...
        ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
        qk_norm: Optional[str] = None,
        # VLA parameters
        training_phase: str = "joint",  # "joint", "action", "traj2d", "action_mini"
        # action head parameters
        action_num_blocks:int = 6,
        action_pred_horizon: int = 6,
        action_dim: int = 7,
        action_hidden_dim: int = 1536,
        action_nheads: int = 24,
        # action_attention_head_dim: Optional[int] = 64,
        n_conditions: int = 0,
        token_fusion: str = "concat",  # "concat", "cross"
        resnet_store_path: str = "cache/visual_features/resnet18/IN_1M_resnet18.pth",
        odim: int = 32,
        view_feature_dim: int = 1920,
        use_film: bool = False,
        combined_temb: bool = False,
        final_layer_norm: bool = True,
        layerwise_vlm_fusion: bool = False,
        state_drop_prob: float = 0.0,
    ):
        super().__init__()
        # self.inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim

        self.state_drop_prob = state_drop_prob
        self.combined_temb = combined_temb
        if self.combined_temb:
            self.time_ins_embed = CombinedTimestepTextProjEmbeddingsND(
                embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
            ) # used; ND variant also accepts per-token (B,Tp) timesteps for RTC
        else:
            self.time_ins_embed = _TimeNetwork(time_dim=256, out_dim=action_hidden_dim)
        """ self.lang_embedder = nn.Linear(joint_attention_dim, caption_projection_dim) """

        # Set the observation projection
        self.obs_proj = ObservationProjection(
            action_pred_horizon=action_pred_horizon, 
            action_dim=action_dim, 
            hidden_dim=self.inner_dim, # not used
            output_dim=action_hidden_dim,
            n_conditions=n_conditions,
            token_fusion=token_fusion,
            resnet_store_path=resnet_store_path,
            odim=odim,
            view_feature_dim=view_feature_dim,
            use_film=use_film
        )

        total_params = sum(p.numel() for p in self.obs_proj.parameters() if p.requires_grad)
        logger.debug(f"ObservationEncoder parameters: {total_params:,}")
        # Set the action projection
        self.action_proj_in = ActionProjectionIn(
            action_pred_horizon=action_pred_horizon, 
            action_dim=action_dim, 
            output_dim=action_hidden_dim,
        )
        total_params = sum(p.numel() for p in self.action_proj_in.parameters() if p.requires_grad)
        logger.debug(f"ActionProjectionIn parameters: {total_params:,}")

        self.transformer_blocks = nn.ModuleList(
            [
                # IdentityBlock()
                VLATransformerBlock( # JointTransformerBlock
                    dim=action_hidden_dim,
                    num_attention_heads=action_nheads,
                    attention_head_dim=attention_head_dim,
                    # layerwise VLM fusion feeds each block a fresh VLM layer and discards the obs
                    # output, so make every block context_pre_only (obs = KV-only): no obs
                    # FF / to_add_out. Otherwise only the last block is context_pre_only.
                    context_pre_only=(layerwise_vlm_fusion or i == action_num_blocks - 1),
                    qk_norm=qk_norm,
                    # use_dual_attention=True if i in dual_attention_layers else False,
                    training_phase=training_phase,  # "joint", "action", "traj2d", "action_mini"
                )
                for i in range(action_num_blocks)
            ]
        )
        total_params = sum(p.numel() for p in self.transformer_blocks[0].parameters() if p.requires_grad)
        logger.debug(f"ActionTransformerBlock parameters: {total_params:,}")

        self.action_proj_out = ActionProjectionOut(
            hidden_size=action_hidden_dim,
            action_dim=action_dim,
            final_layer_norm=final_layer_norm
        )
        total_params = sum(p.numel() for p in self.action_proj_out.parameters())
        logger.debug(f"ActionProjectionOut parameters: {total_params:,}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        block_controlnet_hidden_states: Optional[List] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        vlm_attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[List[torch.Tensor], ActionTransformerModelOutput]: 
        assert joint_attention_kwargs is not None

        if self.config.n_conditions > 0: # type: ignore
            assert joint_attention_kwargs["traj2ds"].shape[1] == self.config.n_conditions, "model n_conditions does not match" # type: ignore

        # L
        if self.combined_temb:
            temb = self.time_ins_embed(timestep, pooled_projections)
        else: 
            temb = self.time_ins_embed(timestep)

        """ lang_hidden_states = self.lang_embedder(encoder_hidden_states) """

        # A
        noisy_action = joint_attention_kwargs.pop("action_hidden_embeds")
        action_hidden_states = self.action_proj_in(noisy_action)

        # Obs: V+Proprio. views is (B, V, N, D): V==1 -> single last-layer context that
        # evolves through the blocks (default); V==n_blocks -> layerwise VLM fusion, block i gets
        # its own VLM layer as a FRESH context (tokenized per layer, obs not carried).
        views = joint_attention_kwargs["views"]

        # Random state dropout
        obs = joint_attention_kwargs["obs"]
        if self.training and self.state_drop_prob > 0.0:
            keep = (torch.rand(obs.shape[0], device=obs.device) >= self.state_drop_prob)
            obs = obs * keep.view(obs.shape[0], *([1] * (obs.dim() - 1))).to(obs.dtype)

        layerwise_vlm_fusion = views.shape[1] > 1
        if layerwise_vlm_fusion:
            assert views.shape[1] == len(self.transformer_blocks), (
                f"layerwise VLM fusion views has {views.shape[1]} layers != {len(self.transformer_blocks)} blocks"
            )
            per_block_obs = []
            obs_token_mask = None
            for i in range(views.shape[1]):
                obs_i, obs_token_mask = self.obs_proj(
                    views=views[:, i:i + 1],
                    obs=obs,
                    traj2ds=joint_attention_kwargs["traj2ds"],
                    text_embeddings=pooled_projections,
                    vlm_attn_mask=vlm_attn_mask,
                )
                per_block_obs.append(obs_i)
            obs_hidden_states = per_block_obs[0]
        else:
            obs_hidden_states, obs_token_mask = self.obs_proj(
                views=views,
                obs=obs,
                traj2ds=joint_attention_kwargs["traj2ds"],
                text_embeddings=pooled_projections,
                vlm_attn_mask=vlm_attn_mask
            ) # S, B, d_model

        for index_block, block in enumerate(self.transformer_blocks):
            is_skip = True if skip_layers is not None and index_block in skip_layers else False
            if is_skip:
                continue

            # layerwise VLM fusion: feed this block its own VLM layer's fresh context and discard
            # the bidirectionally-updated obs (each block re-anchors to its VLM layer).
            obs_in = per_block_obs[index_block] if layerwise_vlm_fusion else obs_hidden_states
            action_hidden_states, obs_out = block(
                # hidden_states=hidden_states, # V
                # lang_hidden_states=lang_hidden_states, # L
                action_hidden_states=action_hidden_states, # A
                obs_hidden_states=obs_in, # O
                temb=temb,
                obs_token_mask=obs_token_mask
                # time_enc=time_enc,
                # obs_pos_emb=obs_pos_emb,
                # joint_attention_kwargs=joint_attention_kwargs,
            )
            if not layerwise_vlm_fusion:
                obs_hidden_states = obs_out

        action_output = self.action_proj_out(
            x=action_hidden_states,
            t=temb,
            # cond=obs_hidden_states,
        )
        return ActionTransformerModelOutput(action=action_output)

