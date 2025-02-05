import torch
from pathlib import Path


def save_model(model: torch.nn.Module,
               dir: str,
               model_name: str):
    dir_path = Path(dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "Invalid model name\
        must end with .pt or .pth"
    save_path = dir_path / model_name

    print(f"[INFO] Saving model to: {save_path}")
    torch.save(obj=model.state_dict(),
               f=save_path)