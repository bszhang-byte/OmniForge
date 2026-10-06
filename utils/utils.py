from typing import Union
import os
from pathlib import Path
import re
import zipfile

def get_asset_dir() -> Path:
    return (resources.files(__package__.split('.')[0])  / ".." / ".." ).resolve() / "assets" # type: ignore

def resolve_path(path: Union[str, Path], subdir="data", auto_download=False) -> Path:
    if Path(path).absolute().exists():
       return Path(path).absolute()
    
    source_dir = get_asset_dir() / path
    if source_dir.exists():
        return source_dir
    
    if "DATA_HOME" in os.environ and subdir == "data":
        data_dir = Path(os.environ["DATA_HOME"])
        filepath = data_dir / path
        if filepath.exists():
            return filepath
    
    if "PSI_HOME" in os.environ:
        proj_dir = Path(os.environ["PSI_HOME"])
        filepath = proj_dir / subdir / path
        if filepath.exists():
            return filepath
        
    if auto_download:
        # auto download the file from remmote huggingface repo "USC-PSI-Lab/psi-data" and extract to the data dir
        from huggingface_hub import snapshot_download
        repo_id = "USC-PSI-Lab/psi-data"
        filename = path if isinstance(path, str) else path.name
        try:

            def _parse_zip_file_from_rel_path(rel_path: str) -> str:
                pattern = r"^assets/robots/([^/]+)/.+\.urdf$"
                match = re.match(pattern, rel_path)
                if match:
                    robot_name = match.group(1)
                    return f"assets/robots/{robot_name}.zip"
                
                pattern = r"^data/egodex/([^/]+)/.+$"
                match = re.match(pattern, rel_path)
                if match:
                    data_name = match.group(1)
                    return f"data/egodex/{data_name}.zip"

                ... # add more patterns as needed

            zip_file = _parse_zip_file_from_rel_path(filename)
            # print(zip_file);
            # auto download
            snapshot_download(
                repo_id="USC-PSI-Lab/psi-data",
                allow_patterns=[zip_file],
                local_dir=os.getcwd(),
                repo_type="dataset",
                # resume_download=True,
            )
            # print(data_dir);exit(0)
            # zip_path = os.path.join(data_dir, zip_file)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(zip_file))
        except Exception as e:
            print(f"Failed to download {filename} from HuggingFace repo {repo_id}: {e}")
            raise e 
    return Path(path)
