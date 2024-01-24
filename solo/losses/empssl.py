
from typing import Any, List, Sequence, Dict
import torch
import torch.distributed as dist
import torch.nn.functional as F

# CODE ADAPTED FROM https://github.com/tsb0601/EMP-SSL/blob/main/main.py

def cosine_similarity_loss_func(z_list: List[torch.Tensor], z_avg) -> torch.Tensor:
        z_sim = 0
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)
        
        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
        
        z_sim = z_sim/num_patch
                
        return -z_sim


def calculate_TCR(W, eps):
    m, p = W.shape  #[B, d]
    I = torch.eye(p,device=W.device)
    scalar = p / (m * eps)
    logdet = torch.logdet(I + scalar * W.T @ W)
    return -logdet / 2.

def calculate_TCR_for_list(z_list, eps):
    loss_per_z = [calculate_TCR(z, eps) for z in list(z_list)]
    loss = sum(loss_per_z) / len(loss_per_z)
    return loss