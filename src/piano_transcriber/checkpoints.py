from pathlib import Path
import torch
def save_checkpoint(path,model,model_config=None,feature_config=None,extra=None):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(),model_config=model_config or {},
                    feature_config=feature_config or {},extra=extra or {}),path)
def load_checkpoint(path,model,map_location="cpu"):
    obj=torch.load(path,map_location=map_location,weights_only=False)
    model.load_state_dict(obj.get("state_dict",obj)); model.eval(); return obj
