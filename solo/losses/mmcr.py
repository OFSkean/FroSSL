
from typing import Any, List, Sequence, Dict
import torch
import torch.distributed as dist

def maximum_manifold_capacity_loss_func(z_list: List[torch.Tensor]) -> torch.Tensor:
    # normalize repr. along the batch dimension
    normalized_z_list = []
    for z in z_list:
        normalized_z = torch.nn.functional.normalize(z, dim=-1) # (N, D)
        normalized_z_list.append(normalized_z)

    normalized_z_list = torch.stack(normalized_z_list)  #(num_augs, N, D)
    centroids = torch.mean(normalized_z_list, dim=0)    #(N, D)
    loss = -torch.linalg.matrix_norm(centroids, ord='nuc')

    return loss