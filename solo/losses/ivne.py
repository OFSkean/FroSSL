
from typing import Any, List, Sequence, Dict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

def get_vne(H):
    Z = torch.nn.functional.normalize(H, dim=1)
    sing_val = torch.svd(Z / np.sqrt(Z.shape[0]))[1]
    eig_val = sing_val ** 2
    return - (eig_val * torch.log(eig_val)).nansum()

# two view loss
def ivne_loss_func(
    z_a: torch.Tensor, z_b: torch.Tensor, alpha_cos, alpha_vne
) -> torch.Tensor:
    """Computes the IVNE loss."""

    a_vne = get_vne(z_a)
    b_vne = get_vne(z_b)
    weighted_vne = alpha_vne * 0.5 * (a_vne + b_vne)

    weighted_cossim = alpha_cos * F.cosine_similarity(z_a, z_b).mean()

    loss = weighted_cossim + weighted_vne
    return -loss