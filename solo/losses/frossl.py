
from typing import Any, List, Sequence, Dict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import repitl.kernel_utils as ku

# two view loss
def frossl_loss_func(
    z_a: torch.Tensor, z_b: torch.Tensor, kernel_type: str, alpha: float, invariance_weight=1, logger=None
) -> torch.Tensor:
    # normalize repr. along the batch dimension
    z_a = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
    z_b = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

    N = z_a.size(0)
    D = z_a.size(1)

    # # scale down to sqrt(d) sphere
    z_a =  (D**0.5) * z_a / torch.norm(z_a, dim=0)
    z_b =  (D**0.5) * z_b / torch.norm(z_b, dim=0)

    # calculate mse loss
    mse_loss = torch.nn.MSELoss()(z_a, z_b)

    # create kernel
    sigma = (z_a.shape[1] / 2) ** 0.5
    if kernel_type == 'gaussian':
        Ka = ku.gaussianKernel(z_a, z_a, sigma)
        Kb = ku.gaussianKernel(z_b, z_b, sigma)
    elif kernel_type == 'linear':
        Ka = (z_a.T @ z_a) / N
        Kb = (z_b.T @ z_b) / N
    else:
        raise NotImplementedError('Kernel type not implemented')

    # calculate entropy loss
    ent_Ka = -2*torch.log(torch.linalg.norm(Ka, ord='fro'))
    ent_Kb = -2*torch.log(torch.linalg.norm(Kb, ord='fro'))
    obj_entropy = ent_Ka + ent_Kb

    loss = -mse_loss*invariance_weight + obj_entropy

    if logger is not None:
        logger(f"entropy", obj_entropy, on_epoch=True, sync_dist=True)
        logger(f"invariance", mse_loss, on_epoch=True, sync_dist=True)

    return -loss

# multi-view loss. same as above but with multiple views
def multiview_frossl_loss_func(
    z_list: List[torch.Tensor], invariance_weight=1, logger=None
) -> torch.Tensor:
    
    N = z_list[0].size(0)
    D = z_list[1].size(1)

    # normalize repr. along the batch dimension
    normalized_z_list = []
    for z in z_list:
        normalized_z =  (D**0.5) * F.normalize(z)
        normalized_z_list.append(normalized_z)

    average_embedding = torch.mean(torch.stack(normalized_z_list), dim=0)

    total_loss = 0
    for view_idx in range(len(normalized_z_list)):
        view_embeddings = normalized_z_list[view_idx]

        if N > D:
            cov = (view_embeddings.T @ view_embeddings) / D
        else:
            cov = (view_embeddings @ view_embeddings.T) / N

        fro_norm = torch.linalg.norm(cov, ord='fro')

        entropy_loss = -2*torch.log(fro_norm) # bring frobenius square outside log

        invariance_loss = torch.nn.MSELoss()(view_embeddings, average_embedding)

        view_loss = -entropy_loss + invariance_loss*invariance_weight
        total_loss += view_loss

        if logger is not None and view_idx == 0:
            logger(f"fro_norm_{view_idx}", entropy_loss, sync_dist=True)
            logger(f"entropy_{view_idx}", entropy_loss, sync_dist=True)
            logger(f"invariance_{view_idx}", invariance_loss, sync_dist=True)

    return total_loss