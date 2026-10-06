from data.egodex import EgoDexDataset

def build_dataset(cfg, transform=None):
    family=getattr(cfg,"dataset_family","egodex")
    if family in ("egodex","ego_dex"):
        return EgoDexDataset(cfg, transform=transform)
    raise ValueError(f"Unknown dataset_family: {family}. Expected egodex")
