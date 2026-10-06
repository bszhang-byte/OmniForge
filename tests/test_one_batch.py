import sys, pathlib; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import os, torch
from data.egodex import EgoDexDataset
from config.config import Config
def test_one_batch():
    c=Config(root_dir=os.getenv("EGODEX_ROOT","/data1/code/dlx/psi_home/data/egodex"),split="part1",chunk_size=1,upsample_rate=3,frame_index=0,max_episodes=2)
    x=EgoDexDataset(c)[0]
    assert x["actions"].shape[-1]==48 and torch.isfinite(torch.as_tensor(x["actions"])).all()
if __name__=="__main__": test_one_batch(); print("one batch ok")
