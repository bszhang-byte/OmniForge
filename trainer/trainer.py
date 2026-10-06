"""
定位：控制训练循环的类
作用：
Trainer类会被传入的config文件初始化，然后被train.py这一脚本调用train函数，知道整个训练过程结束

注：
    1.Trainer要求模型model能够正确返回loss，无论是何种loss。建议模型model提供compute_loss()这一函数；如果采用其他返回loss的方式，
需要在Trainer类（也即本文件）的loss_from_batch函数中补充好，因为Trainer总是通过loss_from_batch函数得到模型吃入当前batch后的loss
    2.save_checkpoint函数提供了保存ckpt的方法
"""

from __future__ import annotations
from pathlib import Path
import json, torch
from torch.nn.utils import clip_grad_norm_
class Trainer:
    def __init__(self,model,optimizer,train_loader,config,scheduler=None,device=None):
        self.model,self.optimizer,self.loader=model,optimizer,train_loader
        self.cfg,self.scheduler=config,scheduler
        self.device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device); self.step=0
    def _move(self,x):
        if torch.is_tensor(x): return x.to(self.device,non_blocking=True)
        if isinstance(x,dict): return {k:self._move(v) for k,v in x.items()}
        return x
    def loss_from_batch(self,batch):
        target = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(target, "compute_loss"): return target.compute_loss(batch)
        out=self.model(**{k:v for k,v in batch.items() if torch.is_tensor(v)})
        if hasattr(out,"loss"): return out.loss
        if torch.is_tensor(out): return out.mean()
        raise TypeError("model must expose compute_loss(batch), loss, or tensor output")
    def save_checkpoint(self,path=None):
        p=Path(path or self.cfg.output_dir)/f"checkpoint-{self.step}"; p.mkdir(parents=True,exist_ok=True)
        torch.save({"step":self.step,"model":self.model.state_dict(),"optimizer":self.optimizer.state_dict(),"scheduler":self.scheduler.state_dict() if self.scheduler else None},p/"state.pt")
        (p/"config.json").write_text(json.dumps(self.cfg.to_dict(),default=str,indent=2)); return p
    def load_checkpoint(self,path):
        s=torch.load(Path(path)/"state.pt",map_location=self.device)
        self.model.load_state_dict(s["model"]); self.optimizer.load_state_dict(s["optimizer"])
        if self.scheduler and s.get("scheduler"): self.scheduler.load_state_dict(s["scheduler"])
        self.step=int(s.get("step",0))
    def train(self):
        self.model.train()
        acc=max(1,self.cfg.gradient_accumulation_steps)
        self.optimizer.zero_grad(set_to_none=True)
        it=iter(self.loader)
        while self.step<self.cfg.max_training_steps:
            try: batch=next(it)
            except StopIteration: it=iter(self.loader); batch=next(it)
            loss=self.loss_from_batch(self._move(batch))/acc; loss.backward()
            if (self.step+1)%acc==0:
                if self.cfg.max_grad_norm: clip_grad_norm_(self.model.parameters(),self.cfg.max_grad_norm)
                self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler: self.scheduler.step()
            self.step+=1
            if self.step%self.cfg.log_freq==0 or self.step==1: print(f"step={self.step} loss={float(loss.detach()*acc):.6f}",flush=True)
            if self.cfg.checkpointing_steps and self.step%self.cfg.checkpointing_steps==0: self.save_checkpoint()
        return self.step
