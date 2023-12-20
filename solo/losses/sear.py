
import torch

import torch.distributed as dist

import repitl.kernel_utils as ku
import repitl.matrix_itl as itl
import repitl.difference_of_entropies as dent

def sear_loss_func(
    z_a: torch.Tensor, z_b: torch.Tensor, kernel_type: str, alpha: float, rank_calculator: str, 
) -> torch.Tensor:
    """Computes SEAR loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    """
    # normalize repr. along the batch dimension
    z_a = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
    z_b = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

    N = z_a.size(0)
    D = z_a.size(1)

    # # scale down to sqrt(d) sphere
    z_a =  (D**0.5) * z_a / torch.norm(z_a, dim=0)
    z_b =  (D**0.5) * z_b / torch.norm(z_b, dim=0)

    # calculate rank loss
    combi = (z_a + z_b) / (2*z_a.shape[0])
    if rank_calculator == 'nuclear':
        Ct = combi.T @ combi
        obj_nuclear = torch.linalg.norm(Ct, ord='nuc')
    elif rank_calculator == 'frobenius':
        obj_nuclear = torch.linalg.norm(combi, ord='fro')**2
    else:
        raise NotImplementedError('Rank type not implemented')

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
    ent_Ka = itl.matrixAlphaEntropy(Ka, alpha=alpha)
    ent_Kb = itl.matrixAlphaEntropy(Kb, alpha=alpha)
    obj_entropy = ent_Ka + ent_Kb

    loss = obj_nuclear + obj_entropy

    return -loss
