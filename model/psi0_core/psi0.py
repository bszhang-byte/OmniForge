"""Psi0 顶层模型：冻结/可训的 Qwen3-VL 主干 + 流匹配动作专家的组装与推理。
本文件由原单文件 psi0.py 拆分而来——
  基础积木  → model/psi0_blocks.py
  联合注意力 → model/psi0_attention.py
  动作专家  → model/psi0_action_head.py
本文件保留 Psi0Model（顶层组装/训练前向/推理与 RTC 采样），并全量 re-export
拆分前的所有公共名字，旧的 `from model.psi0 import X` 写法不受影响。
历史原单文件已拆分。
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Union
from dataclasses import dataclass
from transformers import Qwen3VLForConditionalGeneration, AutoConfig, AutoProcessor
from PIL import Image
from qwen_vl_utils import process_vision_info
from diffusers.utils.outputs import BaseOutput
from diffusers.utils import logging
logger = logging.get_logger(__name__)

from utils.overwatch import initialize_overwatch
# 类型仅注解用，运行时不导入 config（避免模型层依赖配置层）
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config.config import LaunchConfig

# re-export 拆分前的全部公共名字
from model.psi0_core.psi0_blocks import (CombinedTimestepTextProjEmbeddingsND, FilmConditioning,
    _TimeNetwork, _PositionalEncoding, ActionTransformerModelOutput,
    AdaLayerNormZero, AdaLayerNormContinuous, RMSNorm, HumanFoundationModelOutput)
from model.psi0_core.psi0_attention import JointVLAAttnProcessor, VLATransformerBlock
from model.psi0_core.psi0_action_head import (ObservationProjection, ActionProjectionIn,
    ActionProjectionOut, ActionTransformerModel)

QWEN3VL_VARIANT = "Qwen/Qwen3-VL-2B-Instruct"

overwatch = initialize_overwatch(__name__)

class Psi0Model(nn.Module):
    def __init__(self, model_cfg, vlm_model: Qwen3VLForConditionalGeneration):
        super().__init__()
        
        assert model_cfg.use_dit == False
        self.action_header = ActionTransformerModel(
                resnet_store_path=model_cfg.resnet_store_path,
                odim=model_cfg.odim,
                action_dim=model_cfg.action_dim,
                action_pred_horizon=model_cfg.action_chunk_size,
                view_feature_dim=model_cfg.view_feature_dim,
                use_film=model_cfg.use_film,
                combined_temb=model_cfg.combined_temb,
                action_hidden_dim=model_cfg.hidden_dim,
                action_num_blocks=model_cfg.num_blocks,
                final_layer_norm=model_cfg.final_layer_norm,
                qk_norm=model_cfg.qk_norm,
                pooled_projection_dim=model_cfg.pooled_projection_dim,
                layerwise_vlm_fusion=model_cfg.vlm_layer_indices is not None,
                state_drop_prob=model_cfg.state_drop_prob,
        )

        total_params = sum(p.numel() for p in self.action_header.parameters())
        overwatch.info(f"Total ActionTransformerModel parameters: {total_params:,}")

        self.vlm_model = vlm_model
        total_params = sum(p.numel() for p in self.vlm_model.parameters())
        overwatch.info(f"Total VLM Backbone parameters: {total_params:,}")

        # Depth-matched conditioning: one VLM layer per action block. None -> last layer only.
        self.vlm_layer_indices = getattr(model_cfg, "vlm_layer_indices", None)
        if self.vlm_layer_indices is not None:
            assert not model_cfg.use_dit, "layerwise VLM fusion (vlm_layer_indices) is only supported for ActionTransformerModel (use_dit=False)"
            n_blocks = len(self.action_header.transformer_blocks)
            assert len(self.vlm_layer_indices) == n_blocks, (
                f"vlm_layer_indices has {len(self.vlm_layer_indices)} entries but the action "
                f"header has {n_blocks} blocks; they must match (one VLM layer per block)"
            )
            overwatch.info(f"Depth-matched VLM conditioning: block->layer map {self.vlm_layer_indices}")

    # load pretrained vlm+action head, and all the modules needed for predict_action
    @classmethod
    def from_pretrained(cls, run_dir, ckpt_step, launch_config:LaunchConfig, device):
        if not os.path.exists(run_dir):
            raise ValueError(f"Pretrained model path {run_dir} does not exist.")

        from safetensors.torch import load_file
        file_path = f"{run_dir}/checkpoints/ckpt_{ckpt_step}/model.safetensors"
        if not os.path.exists(file_path):
            raise ValueError(f"Checkpoint file {file_path} does not exist.")
        state_dict = load_file(file_path, device="cpu")

        # init empty vlm backbone from config only (skip loading base pretrained weights)
        vlm_config = AutoConfig.from_pretrained(QWEN3VL_VARIANT)
        # Attention backend: flash_attention_2 on CUDA (if installed), sdpa on XPU/CPU.
        try:
            from transformers.utils import is_flash_attn_2_available
            vlm_config._attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        except Exception:
            vlm_config._attn_implementation = "sdpa"
        vlm_config.dtype = torch.bfloat16
        vlm_config.vision_config.dtype = torch.bfloat16
        vlm_config.text_config.dtype = torch.bfloat16
        vlm_model = Qwen3VLForConditionalGeneration(vlm_config)
        vlm_model = vlm_model.to(dtype=torch.bfloat16) # type: ignore

        vlm_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("action_header."):
                continue
            elif k.startswith('vlm_model.'):
                vlm_state_dict[k.replace("vlm_model.", "")] = v
            else:
                assert False, "check here"
        vlm_state_dict["lm_head.weight"] = vlm_state_dict["model.language_model.embed_tokens.weight"] # TODO check here

        if vlm_state_dict["lm_head.weight"].shape[0] != vlm_model.lm_head.weight.shape[0]:
            vlm_model.resize_token_embeddings(
                vlm_state_dict["lm_head.weight"].shape[0], 
                pad_to_multiple_of = 192,
                mean_resizing = True
            )
            overwatch.info(f"Resized model token embeddings to {vlm_model.lm_head.weight.shape[0]}")

        vlm_model.load_state_dict(vlm_state_dict, strict=True)
        overwatch.info("loaded vlm_backbone checkpoint successfully.")

        # init hfm-together model with vlm backbone
        model = Psi0Model(
            model_cfg=launch_config.model,
            vlm_model = vlm_model,
        )

        # load action head
        action_head_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("vlm_model."):
                continue
            elif k.startswith('action_header.'):
                action_head_state_dict[k.replace("action_header.", "")] = v
            else:
                assert False, "check here"
        model.action_header.load_state_dict(action_head_state_dict, strict=True)
        overwatch.info("loaded action head checkpoint successfully.")

        # load necessary modules
        model.vlm_processor = AutoProcessor.from_pretrained(QWEN3VL_VARIANT)
        # model.tokenizer = model.vlm_processor.tokenizer

        if launch_config.model.noise_scheduler == "ddpm":
                from diffusers.schedulers.scheduling_ddim import DDIMScheduler
                scheduler = DDIMScheduler(
                    num_train_timesteps=launch_config.model.train_diffusion_steps,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule="squaredcos_cap_v2",
                    clip_sample=True,   
                    set_alpha_to_one=True,
                    steps_offset=0,
                    prediction_type="epsilon",
                )
        elif launch_config.model.noise_scheduler == "flow":
            from diffusers.schedulers.scheduling_flow_match_euler_discrete import (FlowMatchEulerDiscreteScheduler)
            scheduler= FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=launch_config.model.train_diffusion_steps, # MUST be 1000 as per pretrained SD3
            )
        else:
            raise ValueError(f"Unsupported noise scheduler: {launch_config.model.noise_scheduler}")

        model.noise_scheduler = scheduler
        model.action_horizon = launch_config.model.action_chunk_size
        model.action_dim = launch_config.model.action_dim
        model.device = device
        return model
    
    def _select_vlm_views(self, all_hidden_states) -> torch.Tensor:
        """Stack the VLM hidden states that condition the action header.

        Default (vlm_layer_indices is None): just the last layer -> (B, 1, N, D), fed to
        every action block (current behaviour). Depth-matched: one VLM layer per action
        block -> (B, n_blocks, N, D), so block i attends to VLM layer vlm_layer_indices[i]
        (openpi-style progressive features). `all_hidden_states` is the VLM's
        output_hidden_states tuple (index 0 = embeddings, 1..L = layers, -1 = last).
        """
        if self.vlm_layer_indices is None:
            return all_hidden_states[-1].unsqueeze(1)
        return torch.stack([all_hidden_states[i] for i in self.vlm_layer_indices], dim=1)

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        action_samples: torch.Tensor,
        states: torch.Tensor,
        timestep: torch.LongTensor,
        traj2ds: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[List[torch.Tensor], HumanFoundationModelOutput]:

        # extract vision + language features (keep all layers for layerwise VLM fusion conditioning)
        all_hidden_states = self.vlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True
        ).hidden_states
        vlm_hidden_states = all_hidden_states[-1]     # last layer, for logging (norm)
        views = self._select_vlm_views(all_hidden_states)  # (B, V, N, D): V=1 or n_blocks

        # Mean per-token L2 norm of the VLM feature right before it is handed to the
        # action header, for logging/monitoring. Detached + fp32; masked to valid tokens
        # (pads carry non-zero norm) when the attention mask lines up.
        with torch.no_grad():
            tok_norm = vlm_hidden_states.detach().float().norm(dim=-1)  # (B, N)
            if attention_mask is not None and attention_mask.shape == tok_norm.shape:
                m = attention_mask.to(tok_norm.dtype)
                vlm_feat_norm = (tok_norm * m).sum() / m.sum().clamp_min(1.0)
            else:
                vlm_feat_norm = tok_norm.mean()

        model_output = self.action_header(
            hidden_states=None,
            timestep=timestep,
            pooled_projections=pooled_projections,  # (B, pooled_projection_dim), for combined_temb
            joint_attention_kwargs=dict(
                action_hidden_embeds=action_samples, # (B,Tp,Da)
                views=views,  # (B,V,N,D)  V=1 (last layer) or n_blocks (layerwise VLM fusion)
                obs=states,  # (B,1,M)
                traj2ds=traj2ds,  # (B, C, 3, H, W)
            ),
            vlm_attn_mask=attention_mask, # (B, seq_len)
            return_dict=return_dict,
        )
        return HumanFoundationModelOutput(action=model_output.action, vlm_feat_norm=vlm_feat_norm)

    def _collate_vlm_batch(self, batch_input_ids: List[torch.Tensor]):
        """Right-pad mixed-length samples like training's PaddedCollatorForTogether
        (trainers/qwen3vl_mixin.py); the mask must also reach the action header as
        `vlm_attn_mask` or it attends over the pads. Padding shifts the obs token's position
        index, so a batched sample is not bit-identical to a solo one (measured harmless).
        """
        pad_id = self.vlm_processor.tokenizer.pad_token_id
        assert pad_id is not None, "tokenizer has no pad_token_id; cannot pad a mixed batch"
        input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_id)
        return input_ids, input_ids.ne(pad_id)

    @torch.inference_mode()
    def predict_action(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: torch.Tensor, # (B, Ts, Ds)
        instructions: List[str], # (B,)
        num_inference_steps: int,
        traj2ds,
        goal_images: List[List[Image.Image]] | None = None,  # B * List of PIL goal images
        pooled_projections: Optional[torch.Tensor] = None,  # (B, D) for combined_temb
        **kwargs: str
    ) -> torch.Tensor:

        bsz = states.shape[0]
        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []
        batch_image_grid_thw = []


        for b, (observation, instruction) in enumerate(zip(observations, instructions)):
            messages = []
            content = [{"type": "image", "image": img} for img in observation]
            content.append({"type": "text", "text": instruction})
            # Match training (PsixModelTransform.build_qwenvl_inputs): goal image(s)
            # are appended AFTER the instruction text.
            if goal_images is not None and goal_images[b] is not None:
                content.extend({"type": "image", "image": img} for img in goal_images[b])
            user_msg = {"role": "user", "content": content}
            messages.append([user_msg])
            texts = [
                self.vlm_processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            inputs = self.vlm_processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            # input_ids torch.Size([1, 80])
            # attention_mask torch.Size([1, 80])
            # pixel_values torch.Size([256, 1536])
            # image_grid_thw torch.Size([1, 3])
            batch_input_ids.append(inputs['input_ids'].squeeze(0))
            batch_attention_mask.append(inputs['attention_mask'].squeeze(0))
            batch_pixel_values.append(inputs['pixel_values'])
            batch_image_grid_thw.append(inputs['image_grid_thw'])

        # Per-sample tokenization means different instructions -> different lengths, so stack()
        # would fail. Right-pad exactly like training's collator (see _collate_vlm_batch).
        batch_input_ids, batch_attention_mask = self._collate_vlm_batch(batch_input_ids)
        # Qwen expects pixel_values / image_grid_thw flattened across ALL images in the
        # batch (no leading batch dim). With >1 image per sample (ego + goal) `stack`
        # would make image_grid_thw 3-D -> thw[:, 2] crashes. `cat` keeps them 2-D:
        #   pixel_values   -> (total_image_patches, 1536)
        #   image_grid_thw -> (total_images, 3)
        batch_pixel_values = torch.cat(batch_pixel_values, dim=0)
        batch_image_grid_thw = torch.cat(batch_image_grid_thw, dim=0)

        with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
            # extract vision + language features
            output = self.vlm_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pixel_values=batch_pixel_values,
                image_grid_thw=batch_image_grid_thw,
                output_hidden_states=True,
                return_dict=True
            )
            vlm_hidden_states_ = output.hidden_states # len(vlm_hidden_states_) == 29
            

            # use hidden states from the last layer
            vlm_hidden_states = vlm_hidden_states_[-1] # shape (B, seq_len, D_h)  shape(16, 80, 2048)
            vlm_hidden_states = vlm_hidden_states.unsqueeze(1) # shape (B, 1, seq_len, D_h) (16, 1, 80, 2048)

            # generate action from noise
            action_samples = torch.randn(
                bsz, self.action_horizon, self.action_dim, device=self.device
            )
            self.noise_scheduler.set_timesteps(num_inference_steps)

            for timestep in self.noise_scheduler.timesteps:
                batched_timestep = timestep.expand(bsz).to(self.device)
                model_pred = self.action_header(
                    hidden_states=None,
                    timestep=batched_timestep,
                    pooled_projections=pooled_projections,  # (B, D) for combined_temb
                    joint_attention_kwargs=dict(
                        action_hidden_embeds=action_samples, # (B,Tp,Da)
                        views=self._select_vlm_views(vlm_hidden_states_),  # (B,V,N,D)
                        obs=states,  # (B,1,M)
                        traj2ds=traj2ds,  # (B, C, 3, H, W)
                    ),
                    # match training: mask the pad positions out of the action header's
                    # cross-attention, otherwise it reads pad-token hidden states
                    vlm_attn_mask=batch_attention_mask,  # (B, seq_len)
                    return_dict=True,
                ).action
                action_samples = self.noise_scheduler.step(
                    model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
                ).prev_sample

        return action_samples.float()

    @torch.inference_mode()
    def predict_action_with_training_rtc_flow(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: torch.Tensor, # (B, Ts, Ds)
        instructions: List[str], # (B,)
        num_inference_steps: int,
        traj2ds, 
        prev_actions: torch.Tensor = None, # (B, H, D) — one previous chunk per sample
        inference_delay: int = 0,
        max_delay: int = 0,
        goal_images: List[List[Image.Image]] | None = None,  # B * List of PIL goal images
        pooled_projections: Optional[torch.Tensor] = None,  # (B, D) for combined_temb
        **kwargs: str
    ) -> torch.Tensor:

        ## RTC related ##
        H = self.action_horizon
        assert prev_actions is not None and inference_delay > 0 and max_delay > 0, "prev_actions, inference_delay and max_delay must be provided"
        # One previous chunk per sample. Deliberately NOT allowing a (1, H, D) broadcast at
        # bsz > 1: that would silently condition every client in a batch on one client's
        # previous chunk. Single-client callers have states.shape[0] == 1 and are unaffected.
        assert prev_actions.shape[0] == states.shape[0], (
            f"prev_actions batch {prev_actions.shape[0]} != states batch {states.shape[0]}")
        prev_actions = prev_actions.to(device=self.device, dtype=torch.float32)

        # Create soft mask for inpainting
        d = inference_delay

        # Validate constraint from paper: d ≤ s ≤ H - d
        assert d < H, f"Constraint violated: d={d}, H={H}. Need d < H"
        # assert d < max_delay, f"Constraint violated: d={d}, max_delay={max_delay}. Need d < max_delay"
        d_limit = min(H, max_delay)
        if d >= d_limit:
            overwatch.warning(
                f"RTC inference_delay d={d} >= limit {d_limit}=min(H={H}, max_delay={max_delay}); "
                f"clamping to {d_limit - 1} — inference is slower than the delay budget"
            )
            d = d_limit - 1

        bsz = states.shape[0]
        # (B, H): d is shared by all samples, but expand explicitly — `batched_timestep_masked`
        # below is derived from this mask and must carry the batch dim, not rely on broadcasting.
        prefix_mask = (torch.arange(H, device=self.device)[None, :] < d).expand(bsz, -1)

        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []
        batch_image_grid_thw = []


        for b, (observation, instruction) in enumerate(zip(observations, instructions)):
            messages = []
            content = [{"type": "image", "image": img} for img in observation]
            content.append({"type": "text", "text": instruction})
            # Match training (PsixModelTransform.build_qwenvl_inputs): goal image(s)
            # are appended AFTER the instruction text.
            if goal_images is not None and goal_images[b] is not None:
                content.extend({"type": "image", "image": img} for img in goal_images[b])
            user_msg = {"role": "user", "content": content}
            messages.append([user_msg])
            texts = [
                self.vlm_processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            inputs = self.vlm_processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            # input_ids torch.Size([1, 80])
            # attention_mask torch.Size([1, 80])
            # pixel_values torch.Size([256, 1536])
            # image_grid_thw torch.Size([1, 3])
            batch_input_ids.append(inputs['input_ids'].squeeze(0))
            batch_attention_mask.append(inputs['attention_mask'].squeeze(0))
            batch_pixel_values.append(inputs['pixel_values'])
            batch_image_grid_thw.append(inputs['image_grid_thw'])

        # Per-sample tokenization means different instructions -> different lengths, so stack()
        # would fail. Right-pad exactly like training's collator (see _collate_vlm_batch).
        batch_input_ids, batch_attention_mask = self._collate_vlm_batch(batch_input_ids)
        # Qwen expects pixel_values / image_grid_thw flattened across ALL images in the
        # batch (no leading batch dim). With >1 image per sample (ego + goal) `stack`
        # would make image_grid_thw 3-D -> thw[:, 2] crashes. `cat` keeps them 2-D:
        #   pixel_values   -> (total_image_patches, 1536)
        #   image_grid_thw -> (total_images, 3)
        batch_pixel_values = torch.cat(batch_pixel_values, dim=0)
        batch_image_grid_thw = torch.cat(batch_image_grid_thw, dim=0)

        with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
            # extract vision + language features
            output = self.vlm_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pixel_values=batch_pixel_values,
                image_grid_thw=batch_image_grid_thw,
                output_hidden_states=True,
                return_dict=True
            )
            vlm_hidden_states_ = output.hidden_states # len(vlm_hidden_states_) == 29
            

            # use hidden states from the last layer
            vlm_hidden_states = vlm_hidden_states_[-1] # shape (B, seq_len, D_h)  shape(16, 80, 2048)
            vlm_hidden_states = vlm_hidden_states.unsqueeze(1) # shape (B, 1, seq_len, D_h) (16, 1, 80, 2048)

            # generate action from noise
            action_samples = torch.randn(
                bsz, self.action_horizon, self.action_dim, device=self.device
            )
            self.noise_scheduler.set_timesteps(num_inference_steps)

            # self.noise_scheduler.timesteps: tensor([1000.,  889.,  778.,  667.,  556.,  445.,  334.,  223.,  112.,    1.])
            # self.noise_scheduler.sigmas: tensor([1.0000, 0.8890, 0.7780, 0.6670, 0.5560, 0.4450, 0.3340, 0.2230, 0.1120, 0.0010, 0.0000])

            for i, timestep in enumerate(self.noise_scheduler.timesteps):
                # batched_timestep = timestep.expand(bsz).to(self.device).detach()

                batched_timestep_masked = torch.where(prefix_mask, 0, timestep.to(self.device)) # shape (B, H)

                # replace action_samples with clean prev_actions when prefix_mask == True
                action_samples = torch.where(prefix_mask[:, :, None], prev_actions, action_samples)

                model_pred = self.action_header(
                    hidden_states=None,
                    timestep=batched_timestep_masked,
                    pooled_projections=pooled_projections,  # (B, D) for combined_temb
                    joint_attention_kwargs=dict(
                        action_hidden_embeds=action_samples, # (B,Tp,Da)
                        views=self._select_vlm_views(vlm_hidden_states_),  # (B,V,N,D)
                        obs=states,  # (B,1,M)
                        traj2ds=traj2ds,  # (B, C, 3, H, W)
                    ),
                    # match training: mask the pad positions out of the action header's
                    # cross-attention, otherwise it reads pad-token hidden states
                    vlm_attn_mask=batch_attention_mask,  # (B, seq_len)
                    return_dict=True,
                ).action

                action_samples = self.noise_scheduler.step(
                    model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
                ).prev_sample

                # if i == len(self.noise_scheduler.timesteps) - 1:
                #     action_samples = torch.where(prefix_mask[:, :, None], prev_actions, action_samples)

        return action_samples.float()

    @torch.no_grad()
    def predict_action_with_rtc_flow(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: torch.Tensor, # (B, Ts, Ds)
        instructions: List[str], # (B,)
        num_inference_steps: int,
        traj2ds, 
        prev_actions: torch.Tensor = None, # (1, H, D)
        inference_delay: int = 0,
        # s: free steps at the tail. None = derive H-2d. Matches
        # psix.predict_action_with_test_time_rtc -- and must stay `is not None`
        # below, so that an explicit s=0 is honoured rather than silently derived.
        execution_horizon: int | None = None,
        pooled_projections: Optional[torch.Tensor] = None,  # (B, D) for combined_temb
        mask_schedule: str = "exponential",
        guidance_weight: float = 5.0,  # DEPRECATED, see the guidance scaling note below
        guidance_alpha: float = 0.9,
        **kwargs: str
    ) -> np.ndarray:

        ## RTC related ##
        H = self.action_horizon
        assert prev_actions is not None, "prev_actions must be provided"
        assert prev_actions.shape[0] == 1
        assert tuple(prev_actions.shape) == (1, H, self.action_dim), (
            f"prev_actions shape {tuple(prev_actions.shape)} != (1,{H},{self.action_dim}); "
            "the caller must shift/pad the previous chunk to length H")
        prev_actions = prev_actions.to(device=self.device, dtype=torch.float32)

        # Create soft mask for inpainting
        d = int(inference_delay)
        s = int(execution_horizon) if execution_horizon is not None else max(1, H - 2 * d)

        # Paper constraint d ≤ s ≤ H-d. Clamp instead of asserting (matches
        # psix.predict_action_with_test_time_rtc): a server that hands over a
        # d/s pair one tick outside the window should degrade to the nearest
        # valid mask, not kill the request mid-episode. d=0 is legal here -- it
        # just means nothing is hard-frozen; rows [0, H-s) still carry the ramp,
        # and only s == H reduces this call to plain predict_action.
        d = max(0, min(d, H - 1))
        s = max(d, min(s, H - d))

        # Create soft mask [H] according to Equation 5
        mask = self._create_soft_mask(
            H, d, s, schedule=mask_schedule, device=self.device
        )
        mask_expanded = mask.view(1, -1, 1).expand(1, self.action_horizon, self.action_dim)

        bsz = states.shape[0]
        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []
        batch_image_grid_thw = []

        for observation, instruction in zip(observations, instructions):
            messages = []
            content = [{"type": "image", "image": img} for img in observation]
            content.append({"type": "text", "text": instruction})
            user_msg = {"role": "user", "content": content}
            messages.append([user_msg])
            texts = [
                self.vlm_processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True
                )
                for m in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            inputs = self.vlm_processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            batch_input_ids.append(inputs['input_ids'].squeeze(0))
            batch_attention_mask.append(inputs['attention_mask'].squeeze(0))
            batch_pixel_values.append(inputs['pixel_values'])
            batch_image_grid_thw.append(inputs['image_grid_thw'])

        # Same collation as predict_action / predict_action_with_training_rtc_flow.
        # This path used to torch.stack() everything, which gave pixel_values a
        # leading batch dim (B, patches, 1536) -- a shape Qwen does not accept --
        # and could not pad ragged prompts at all. Keep the three sampling paths
        # byte-identical up to the guidance term, or a test-time RTC rollout is
        # not comparable with the open-loop one.
        batch_input_ids, batch_attention_mask = self._collate_vlm_batch(batch_input_ids)
        batch_pixel_values = torch.cat(batch_pixel_values, dim=0)
        batch_image_grid_thw = torch.cat(batch_image_grid_thw, dim=0)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # extract vision + language features
            output = self.vlm_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pixel_values=batch_pixel_values,
                image_grid_thw=batch_image_grid_thw,
                output_hidden_states=True,
                return_dict=True
            )
            vlm_hidden_states_ = output.hidden_states # len(vlm_hidden_states_) == 29

            # use hidden states from the last layer
            vlm_hidden_states = vlm_hidden_states_[-1] # shape (B, seq_len, D_h)  shape(16, 80, 2048)
            vlm_hidden_states = vlm_hidden_states.unsqueeze(1) # shape (B, 1, seq_len, D_h) (16, 1, 80, 2048)

            # generate action from noise
            action_samples = torch.randn(
                bsz, self.action_horizon, self.action_dim, device=self.device
            )
            # target_noise = action_samples.clone().detach()
            self.noise_scheduler.set_timesteps(num_inference_steps)

            for timestep in self.noise_scheduler.timesteps:
                batched_timestep = timestep.expand(bsz).to(self.device).detach()

                with torch.enable_grad():
                    # Detach and enable gradient only for sample_actions in this step
                    action_samples_grad = action_samples.detach().requires_grad_(True)

                    model_pred = self.action_header(
                        hidden_states=None,
                        timestep=batched_timestep,
                        pooled_projections=pooled_projections,  # (B, D) for combined_temb
                        joint_attention_kwargs=dict(
                            action_hidden_embeds=action_samples_grad, # (B,Tp,Da)
                            views=self._select_vlm_views(vlm_hidden_states_),  # (B,V,N,D)
                            obs=states,  # (B,1,M)
                            traj2ds=traj2ds,  # (B, C, 3, H, W)
                        ),
                        # match training: mask the pad positions out of the action
                        # header's cross-attention, otherwise it reads pad-token
                        # hidden states
                        vlm_attn_mask=batch_attention_mask,  # (B, seq_len)
                        return_dict=True,
                    ).action

                    # 1. 外推干净data X_0
                    tau = self.noise_scheduler.sigmas[self.noise_scheduler.index_for_timestep(timestep)].detach()
                    pred_x0 = action_samples_grad - tau * model_pred
                    # 2. 计算(Y - pred_X_0) * weights
                    error = (prev_actions - pred_x0.detach()) * mask_expanded
                    # 3. vjp
                    pinv_correction = torch.autograd.grad(
                        outputs=pred_x0,  # [1,T,D]
                        inputs=action_samples_grad,        # [1,T,D]
                        grad_outputs=error,         # 匹配形状 [1,T,D]
                        retain_graph=False,
                        create_graph=False
                    )[0]  # 输出 [1,T,D]
                    # print("pinv_correction:", pinv_correction)

                # 4. 缩放修正项，修正v
                pinv_norm = pinv_correction.norm()
                if tau.item() >= 1.0 or pinv_norm < 1e-8:
                    # tau == 1: the sample is still pure noise, there is nothing to guide yet
                    correction = torch.zeros_like(pinv_correction)
                else:
                    scale = guidance_alpha * error.norm() / (tau * pinv_norm)
                    correction = scale * pinv_correction

                model_pred = model_pred.detach() - correction

                action_samples = self.noise_scheduler.step(
                    model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
                ).prev_sample

        return action_samples.float()

    def predict_action_with_rtc_flow_naive_inpaint(
        self,
        observations: List[List[Image.Image]],  # B * List of PIL Image as [view1, view2]
        states: torch.Tensor, # (B, Ts, Ds)
        instructions: List[str], # (B,)
        num_inference_steps: int,
        traj2ds, 
        prev_actions: torch.Tensor = None, # (1, H, D)
        inference_delay: int = 0,
        execution_horizon: int = 0,
        mask_schedule: str = "exponential",
        guidance_weight: float = 5.0,
        **kwargs: str
    ) -> np.ndarray:
        with torch.no_grad():

            ## RTC related ##
            H = self.action_horizon
            assert prev_actions is not None and inference_delay > 0 and execution_horizon > 0, "prev_actions, inference_delay and execution_horizon must be provided"
            assert prev_actions.shape[0] == 1
            prev_actions = prev_actions.to(device=self.device, dtype=torch.float32)

            # Create soft mask for inpainting
            d = inference_delay
            s = execution_horizon

            # Validate constraint from paper: d ≤ s ≤ H - d
            assert d <= s and s <= H - d, f"Constraint violated: d={d}, s={s}, H={H}. Need d ≤ s ≤ H-d"

            # Create soft mask [H] according to Equation 5
            mask = self._create_soft_mask(
                H, d, s, schedule=mask_schedule, device=self.device
            )
            mask_expanded = mask.view(1, -1, 1).expand(1, self.action_horizon, self.action_dim)
            ##              ##


            bsz = states.shape[0]
            batch_input_ids = []
            batch_attention_mask = []
            batch_pixel_values = []
            batch_image_grid_thw = []


            for observation, instruction in zip(observations, instructions):
                messages = []
                content = [{"type": "image", "image": img} for img in observation]
                content.append({"type": "text", "text": instruction})
                user_msg = {"role": "user", "content": content}
                messages.append([user_msg])
                texts = [
                    self.vlm_processor.apply_chat_template(
                        m, tokenize=False, add_generation_prompt=True
                    )
                    for m in messages
                ]
                image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
                inputs = self.vlm_processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                # input_ids torch.Size([1, 80])
                # attention_mask torch.Size([1, 80])
                # pixel_values torch.Size([256, 1536])
                # image_grid_thw torch.Size([1, 3])
                batch_input_ids.append(inputs['input_ids'].squeeze(0))
                batch_attention_mask.append(inputs['attention_mask'].squeeze(0))
                batch_pixel_values.append(inputs['pixel_values'])
                batch_image_grid_thw.append(inputs['image_grid_thw'].squeeze(0))

            batch_input_ids = torch.stack(batch_input_ids) # (B, 80)
            batch_attention_mask = torch.stack(batch_attention_mask) # (B, 80)
            batch_pixel_values = torch.stack(batch_pixel_values) # (B, 256, 1536)
            batch_image_grid_thw = torch.stack(batch_image_grid_thw) # (B, 3)

            with torch.autocast(str(self.device).split(":")[0], dtype=torch.bfloat16):
                # extract vision + language features
                output = self.vlm_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    pixel_values=batch_pixel_values,
                    image_grid_thw=batch_image_grid_thw,
                    output_hidden_states=True,
                    return_dict=True
                )
                vlm_hidden_states_ = output.hidden_states # len(vlm_hidden_states_) == 29
                

                # use hidden states from the last layer
                vlm_hidden_states = vlm_hidden_states_[-1] # shape (B, seq_len, D_h)  shape(16, 80, 2048)
                vlm_hidden_states = vlm_hidden_states.unsqueeze(1) # shape (B, 1, seq_len, D_h) (16, 1, 80, 2048)

                # generate action from noise
                action_samples = torch.randn(
                    bsz, self.action_horizon, self.action_dim, device=self.device
                )
                self.noise_scheduler.set_timesteps(num_inference_steps)

                # Fixed noise for consistency in inpainting
                # Use the same noise as action_samples to ensure the whole sequence starts from the same latent x_1
                target_noise = action_samples.clone()

                for i, timestep in enumerate(self.noise_scheduler.timesteps):
                    batched_timestep = timestep.expand(bsz).to(self.device).detach()

                    ### Naive inpainting (Repaint method):
                    # Step 1: Add noise to prev_actions matching current timestep's noise level
                    # This ensures consistency in noise levels across the entire action sequence
                    tau = self.noise_scheduler.sigmas[self.noise_scheduler.index_for_timestep(timestep)]
                    noisy_prev_actions = (1 - tau) * prev_actions + tau * target_noise
                    
                    # Step 2: Replace first d positions with noisy prev_actions (hard mask)
                    action_samples[:, :d, :] = noisy_prev_actions[:, :d, :]

                    # Step 3: Model prediction
                    model_pred = self.action_header(
                        hidden_states=None,
                        timestep=batched_timestep,
                        joint_attention_kwargs=dict(
                            action_hidden_embeds=action_samples, # (B,Tp,Da)
                            views=self._select_vlm_views(vlm_hidden_states_),  # (B,V,N,D)
                            obs=states,  # (B,1,M)
                            traj2ds=traj2ds,  # (B, C, 3, H, W)
                        ),
                        return_dict=True,
                    ).action

                    # Step 4: Denoise one step (updates all positions including first d)
                    action_samples = self.noise_scheduler.step(
                        model_output=model_pred, timestep=timestep, sample=action_samples # type: ignore
                    ).prev_sample
                    
                    # Step 5: Re-inpaint with noise level matching the denoised state
                    # TODO: do this or not??? boqian
                    if i == len(self.noise_scheduler.timesteps) - 1:
                        # Last step: use clean prev_actions (tau = 0)
                        action_samples[:, :d, :] = prev_actions[:, :d, :]

            return action_samples

    def _create_soft_mask(self, H, d, s, schedule="exponential", device="cpu"):
        """
        Create soft mask for RTC inpainting (Equation 5 in paper).
        
        Paper: "Real-Time Execution of Action Chunking Flow Policies"
        Figure 3 and Equation 5:
        
        W_i = { 1                           if i < d (frozen region)
              { c_i * (e^(c_i) - 1)/(e - 1) if d ≤ i < H - s (intermediate region)
              { 0                           if i ≥ H - s (free region)
        
        where c_i = (H - s - i)/(H - s - d + 1)
        
        Args:
            H: Prediction horizon (total sequence length)
            d: Inference delay (number of frozen steps, already executed)
            s: Execution horizon (non-overlapping steps at end). 
            schedule: 'exponential' (paper default), 'linear', 'hard', or 'simple'
        
        Returns:
            mask: [H] tensor with values in [0, 1]
                  Guidance weights for each timestep
        """
        mask = torch.zeros(H, device=device)
        
        if schedule == "hard":
            # Hard mask: 1 for frozen, 0 for rest
            mask[:d] = 1.0
        
        elif schedule == "linear":
            # Linear decay
            mask[:d] = 1.0
            # Three-region version (full paper)
            overlap_end = H - s
            if d < overlap_end:
                # Linear decay from 1 to 0 in intermediate region
                indices = torch.arange(d, overlap_end, device=device).float()
                mask[d:overlap_end] = 1.0 - (indices - d) / (overlap_end - d)
            # mask[overlap_end:] remains 0
        
        elif schedule == "exponential":
            # Exponential decay (paper default, Equation 5)
            mask[:d] = 1.0
            
            # Three-region version (full paper implementation)
            overlap_end = H - s
            if d < overlap_end:
                indices = torch.arange(d, overlap_end, device=device).float()
                # c_i = (H - s - i) / (H - s - d + 1)
                c_i = (overlap_end - indices) / (overlap_end - d + 1)
                # W_i = c_i * (e^(c_i) - 1) / (e - 1)
                e = torch.tensor(torch.e, device=device)
                mask[d:overlap_end] = c_i * (torch.exp(c_i) - 1) / (e - 1)
            # mask[overlap_end:] remains 0
        
        elif schedule == "simple":
            # Simplified exponential (for backward compatibility)
            mask[:d] = 1.0
            if d < H:
                indices = torch.arange(d, H, device=device).float()
                mask[d:] = torch.exp(-5.0 * (indices - d) / (H - d))
        
        else:
            raise ValueError(f"Unknown mask schedule: {schedule}")
        
        return mask
    
    @torch.inference_mode()
    def chat(self):
        ...
