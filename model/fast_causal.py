from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
TOK_ACTION_START="<|action_start|>"; TOK_ACTION_END="<|action_end|>"
class FastCausalModel(torch.nn.Module):
    """Qwen3-VL causal LM with the official FAST answer-token convention."""
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.processor=AutoProcessor.from_pretrained(cfg.model_name_or_path,local_files_only=cfg.local_files_only)
        self.fast_processor=AutoProcessor.from_pretrained(cfg.pretrained_checkpoint,trust_remote_code=True,local_files_only=True)
        self.fast_processor.action_dim=cfg.action_dim; self.fast_processor.time_horizon=cfg.chunk_size
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.model_name_or_path,dtype=torch.bfloat16 if cfg.mixed_precision=="bf16" else torch.float32,
            local_files_only=cfg.local_files_only,attn_implementation=cfg.attn_implementation)
        tok=self.processor.tokenizer
        # Match upstream FastActionTokenizer: ordinary action tokens first, special tokens second.
        tok.add_tokens([f"<|a_{i}|>" for i in range(cfg.bins)])
        tok.add_special_tokens({"additional_special_tokens":[TOK_ACTION_START,TOK_ACTION_END]})
        self.model.resize_token_embeddings(len(tok))
        self.action_token_begin_idx=tok(f"<|a_0|>")["input_ids"][0]
        if not cfg.tune_mm_vision:
            for p in self.model.visual.parameters(): p.requires_grad=False
        if not cfg.tune_mm_llm:
            for p in self.model.language_model.parameters(): p.requires_grad=False
    def _action_text(self, action):
        ids=self.fast_processor(np.asarray(action,dtype=np.float32))[0]
        return TOK_ACTION_START+"".join(f"<|a_{int(i)}|>" for i in ids)+TOK_ACTION_END
    def _sample_inputs(self, image, instruction, answer):
        image=Image.fromarray(np.asarray(image).astype(np.uint8)).resize((self.cfg.resize[1],self.cfg.resize[0]))
        user={"role":"user","content":[{"type":"image","image":image},{"type":"text","text":str(instruction)}]}
        full=[user,{"role":"assistant","content":[{"type":"text","text":answer}]}]
        user_text=self.processor.apply_chat_template([user],tokenize=False,add_generation_prompt=True)
        full_text=self.processor.apply_chat_template(full,tokenize=False,add_generation_prompt=False)
        ui,uv=process_vision_info([user],image_patch_size=16)
        fi,fv=process_vision_info(full,image_patch_size=16)
        u=self.processor(text=[user_text],images=ui or None,videos=uv or None,padding=True,return_tensors="pt")
        f=self.processor(text=[full_text],images=fi or None,videos=fv or None,padding=True,return_tensors="pt")
        return f,u
    def compute_loss(self,batch):
        imgs=batch["current_images"][0]
        if imgs.ndim==5: imgs=imgs[:,0]
        fulls=[]; prefixes=[]
        for i,instruction in enumerate(batch["instruction"]):
            answer=self._action_text(batch["actions"][i].detach().cpu().numpy())
            f,u=self._sample_inputs(imgs[i],instruction,answer)
            fulls.append(f); prefixes.append(u)
        ids=pad_sequence([x["input_ids"][0] for x in fulls],batch_first=True,padding_value=self.processor.tokenizer.pad_token_id)
        mask=ids.ne(self.processor.tokenizer.pad_token_id)
        labels=ids.clone()
        for i,u in enumerate(prefixes):
            n=int(u["attention_mask"][0].sum().item()); labels[i,:n]=-100
        labels[~mask]=-100
        pixels=torch.cat([x["pixel_values"] for x in fulls],dim=0)
        grids=torch.cat([x["image_grid_thw"] for x in fulls],dim=0)
        dev=self.model.device
        return self.model(input_ids=ids.to(dev),attention_mask=mask.to(dev),pixel_values=pixels.to(dev),image_grid_thw=grids.to(dev),labels=labels.to(dev)).loss

