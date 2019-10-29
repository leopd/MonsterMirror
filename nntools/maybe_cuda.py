import torch

def mbcuda(t: torch.Tensor) -> torch.Tensor:
    """Short for "maybe cuda" this moves a tensor (or model) to cuda if available
    """
    if torch.cuda.is_available():
        return t.cuda()
    else:
        return t

