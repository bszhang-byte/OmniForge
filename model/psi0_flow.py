from __future__ import annotations
from types import SimpleNamespace
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from model.psi0_core.psi0 import Psi0Model

class Psi0FlowAdapter(torch.nn.Module):
    """Training adapter around the packaged Psi0Model flow/action head."""
    def __init__(self, cfg):
        super().__init__(); self.cfg=cfg
        self.processor=AutoProcessor.from_pretrained(cfg.model_name_or_path,local_files_only=cfg.local_files_only)
        vlm=Qwen3VLForConditionalGeneration.from_pretrained(cfg.model_name_or_path,dtype=torch.bfloat16 if cfg.mixed_precision=="bf16" else torch.float32,local_files_only=cfg.local_files_only,attn_implementation=cfg.attn_implementation)
        core_cfg=SimpleNamespace(**cfg.to_dict()); core_cfg.action_chunk_size=cfg.chunk_size
        self.psi0=Psi0Model(core_cfg,vlm)
    def _vlm_inputs(self,images,instructions):
        xs=[]
        for image,instruction in zip(images,instructions):
            pil=Image.fromarray(image.astype("uint8")).resize((self.cfg.resize[1],self.cfg.resize[0]))
            msg=[{"role":"user","content":[{"type":"image","image":pil},{"type":"text","text":str(instruction)}]}]
            text=self.processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
            ii,vv=process_vision_info(msg,image_patch_size=16)
            xs.append(self.processor(text=[text],images=ii or None,videos=vv or None,padding=True,return_tensors="pt"))
        return (torch.cat([x["input_ids"] for x in xs]),torch.cat([x["attention_mask"] for x in xs]),torch.cat([x["pixel_values"] for x in xs]),torch.cat([x["image_grid_thw"] for x in xs]))
    def compute_loss(self,batch):
        imgs=batch["current_images"][0]
        if imgs.ndim==5: imgs=imgs[:,0]
        ids,mask,pixels,grids=self._vlm_inputs(imgs.detach().cpu().numpy(),batch["instruction"])
        dev=self.psi0.vlm_model.device; actions=batch["actions"].to(dev); states=batch["states"].to(dev)
        noise=torch.randn_like(actions); timestep=torch.rand(actions.shape[0],device=dev)
        out=self.psi0(input_ids=ids.to(dev),attention_mask=mask.to(dev),pixel_values=pixels.to(dev),image_grid_thw=grids.to(dev),action_samples=noise,states=states,timestep=timestep)
        return torch.nn.functional.mse_loss(out.action,actions-noise)
