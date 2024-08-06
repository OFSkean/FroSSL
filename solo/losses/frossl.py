
from typing import Any, List, Sequence, Dict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import repitl.kernel_utils as ku

def frossl_loss_func(
    z_a: torch.Tensor, z_b: torch.Tensor, kernel_type: str = "linear", invariance_weight=1, logger=None
) -> torch.Tensor:
    """
    DEPRECATED: use the multiview_frossl_loss_func instead. This is kept to maintain compatibility
        with the original FroSSL paper. In particular, this function uses and 
        batch norm and F.normalize(dim=0) to normalize the representations. The function
        multiview_frossl_loss_func does not use batch norm and uses F.normalize(dim=1) instead.

    Computes the FroSSL loss between two views z_a and z_b.

    Inputs:
        z_a (torch.Tensor): the representation of view a.
        z_b (torch.Tensor): the representation of view b.
        kernel_type (str): the type of kernel to use for the entropy calculation.
            linear kernel is used in the paper
        alpha (float): the alpha parameter for the entropy calculation.
        invariance_weight (float): the weight of the invariance loss.
        logger (Any): the logger to use for logging.
    """
    # normalize repr. along the batch dimension
    z_a = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
    z_b = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

    N = z_a.size(0)
    D = z_a.size(1)

    # scale down to sqrt(d) sphere
    z_a =  (D**0.5) * F.normalize(z_a, p=2, dim=0)
    z_b =  (D**0.5) * F.normalize(z_b, p=2, dim=0)

    # calculate mse loss
    mse_loss = F.mse_loss(z_a, z_b)

    # create kernel
    if kernel_type == 'gaussian':
        sigma = (z_a.shape[1] / 2) ** 0.5
        Ka = ku.gaussianKernel(z_a, z_a, sigma)
        Kb = ku.gaussianKernel(z_b, z_b, sigma)
    elif kernel_type == 'linear':
        if N > D:
            Ka = z_a.T @ z_a
            Kb = z_b.T @ z_b
        else:
            Ka = z_a @ z_a.T
            Kb = z_b @ z_b.T

        Ka = Ka / torch.trace(Ka).item()  # do .item() to avoid float16 casting issues
        Kb = Kb / torch.trace(Kb).item() 
    else:
        raise NotImplementedError('Kernel type not implemented')

    # calculate entropy loss
    # the 2 in the front is to bring the frobenius square outside the log
    ent_Ka = -2*torch.log(torch.linalg.norm(Ka, ord='fro'))
    ent_Kb = -2*torch.log(torch.linalg.norm(Kb, ord='fro'))
    obj_entropy = ent_Ka + ent_Kb

    loss = mse_loss*invariance_weight - obj_entropy

    if logger is not None:
        logger(f"fro_norm_0", torch.linalg.norm(Ka, ord='fro'), sync_dist=True)
        logger(f"entropy_0", ent_Ka, sync_dist=True)
        logger(f"invariance_0", mse_loss/2, sync_dist=True)

    return loss

def multiview_frossl_loss_func(
    z_list: List[torch.Tensor], invariance_weight=1, logger=None
) -> torch.Tensor:
    """
    Computes the FroSSL loss between multiple views.

    Inputs:
        z_list (List[torch.Tensor]): the list of representations of each view.
        invariance_weight (float): the weight of the invariance loss.
        logger (Any): the logger to use

    Outputs:
        torch.Tensor: the total loss
    """ 
    
    N = z_list[0].size(0)
    D = z_list[1].size(1)
    V = len(z_list)

    # normalize repr. along the batch dimension
    normalized_z_list = []
    for z in z_list:
        normalized_z = F.normalize(z, p=2, dim=0)
        normalized_z_list.append(normalized_z)

    average_embedding = torch.mean(torch.stack(normalized_z_list), dim=0)

    total_loss = 0
    for view_idx in range(V):
        view_embeddings = normalized_z_list[view_idx]

        # REGULARIZATION TERM
        if N > D:
            cov = view_embeddings.T @ view_embeddings
        else:
            cov = view_embeddings @ view_embeddings.T
        cov = cov / torch.trace(cov).item()  # do .item() to avoid float16 casting issues
        fro_norm = torch.linalg.norm(cov, ord='fro')
        regularization_loss_term = -2*torch.log(fro_norm) # eq (6) in paper, here we bring frobenius square outside log

        # INVARIANCE TERM
        invariance_loss_term = V * F.mse_loss(view_embeddings, average_embedding) # eq (3) in paper
        invariance_loss_term = invariance_loss_term * D * invariance_weight

        # FROSSL LOSS
        view_loss = -1*regularization_loss_term + invariance_loss_term
        total_loss += view_loss

        if logger is not None and view_idx == 0:
            logger(f"fro_norm_{view_idx}", fro_norm, sync_dist=True)
            logger(f"entropy_{view_idx}", regularization_loss_term, sync_dist=True)
            logger(f"invariance_{view_idx}", invariance_loss_term, sync_dist=True)

    return total_loss