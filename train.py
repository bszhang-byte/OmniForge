"""
定位：整个框架的根启动程序
作用：
1.全部的shell命令都应当用torchrun唤起本程序，本程序识别"--config"命令传入的config文件，创建数据集、优化器、loader、模型、优化器、Trainer；
2.创建完毕后，交由Trainer执行train()函数，完整的训练循环在Trainer.train()函数内部完成

注：
1.保存ckpt的工作亦写在了Trainer类中，所有关于训练循环的改动都应当在Trainer.train()函数中修改，比如将来如果要在每个训练的step中间执行一轮验证，那么就应该在Trainer中修改
2.理论上这个文件不应该再被改动（包括：没有必要导入其他的库）

张博山 2026年9月29日
"""

from __future__ import annotations
import os, json, math
import numpy as np
import torch
from torch.utils.data import DataLoader
from config.config import parse_config
from data.factory import build_dataset
from model.factory import build_model
from trainer.trainer import Trainer

TOK_ACTION_START="<|action_start|>"; TOK_ACTION_END="<|action_end|>"

def build_optimizer(model,cfg):
    decay=[]; nodecay=[]; vision=[]; merger=[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if "visual" in n: (vision if "weight" in n else nodecay).append(p); continue
        if "merger" in n: (merger if "weight" in n else nodecay).append(p); continue
        (decay if p.ndim>=2 and "bias" not in n and "norm" not in n.lower() else nodecay).append(p)
    groups=[]
    if decay: groups.append({"params":decay,"weight_decay":cfg.weight_decay,"lr":cfg.learning_rate})
    if nodecay: groups.append({"params":nodecay,"weight_decay":0.0,"lr":cfg.learning_rate})
    if vision: groups.append({"params":vision,"weight_decay":cfg.weight_decay,"lr":cfg.vision_tower_lr})
    if merger: groups.append({"params":merger,"weight_decay":cfg.weight_decay,"lr":cfg.mm_projector_lr})
    return torch.optim.AdamW(groups,betas=tuple(cfg.betas),eps=cfg.eps)

def build_scheduler(opt,cfg):
    if cfg.lr_scheduler_type=="constant": return torch.optim.lr_scheduler.ConstantLR(opt,factor=1.0,total_iters=1)
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(1,cfg.max_training_steps),eta_min=cfg.min_lr)

def main(argv=None):
    cfg=parse_config(argv); torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    distributed="RANK" in os.environ
    if distributed:
        torch.distributed.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK",0)))
    
    ds=build_dataset(cfg)
    
    sampler=torch.utils.data.distributed.DistributedSampler(ds,shuffle=True) if distributed else None

    loader=DataLoader(ds,batch_size=cfg.train_batch_size,shuffle=sampler is None,sampler=sampler,num_workers=cfg.num_workers,drop_last=True)
    
    model=build_model(cfg)
    opt=build_optimizer(model,cfg)
    sched=build_scheduler(opt,cfg)

    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model=DDP(model,device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,find_unused_parameters=False)
    trainer=Trainer(model,opt,loader,cfg,scheduler=sched)
    if cfg.resume_from_checkpoint: trainer.load_checkpoint(cfg.resume_from_checkpoint)
    trainer.train()

if __name__=="__main__": main()
