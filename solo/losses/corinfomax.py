## adapted from https://github.com/serdarozsoy/corinfomax-ssl/blob/main/cifars-tiny/main_traintest.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Attraction factor of CorInfoMax Loss: MSE loss calculation from outputs of the projection network, z1 (NXD) from 
    the first branch and z2 (NXD) from the second branch. Returns loss part comes from attraction factor (mean squared error).
    """
    return F.mse_loss(z1, z2)



class CorInfoMax_Loss(nn.Module):
    def __init__(self, output_dim, R_ini, la_R, la_mu, R_eps_weight, alpha_tradeoff):
        super().__init__()
        self.R1 = R_ini*torch.eye(output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu1 = torch.zeros(output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.R2 = R_ini*torch.eye(output_dim , dtype=torch.float64, device='cuda', requires_grad=False)
        self.mu2 = torch.zeros(output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.new_R1 = torch.zeros((output_dim, output_dim), dtype=torch.float64, device='cuda', requires_grad=False) 
        self.new_mu1 = torch.zeros(output_dim, dtype=torch.float64, device='cuda', requires_grad=False) 
        self.new_R2 = torch.zeros((output_dim, output_dim), dtype=torch.float64, device='cuda', requires_grad=False) 
        self.new_mu2 = torch.zeros(output_dim, dtype=torch.float64, device='cuda', requires_grad=False) 
        self.la_R = la_R
        self.la_mu = la_mu
        self.R_eps_weight = R_eps_weight
        self.R_eps = self.R_eps_weight*torch.eye(output_dim, dtype=torch.float64, device='cuda', requires_grad=False)
        self.alpha_tradeoff = alpha_tradeoff

        self.invariance_loss = invariance_loss

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        la_R = self.la_R
        la_mu = self.la_mu

        N, D = z1.size()

        # mean estimation
        mu_update1 = torch.mean(z1, 0)
        mu_update2 = torch.mean(z2, 0)
        self.new_mu1 = la_mu*(self.mu1) + (1-la_mu)*(mu_update1)
        self.new_mu2 = la_mu*(self.mu2) + (1-la_mu)*(mu_update2)

        # covariance matrix estimation
        z1_hat =  z1 - self.new_mu1
        z2_hat =  z2 - self.new_mu2
        R1_update = (z1_hat.T @ z1_hat) / N
        R2_update = (z2_hat.T @ z2_hat) / N
        self.new_R1 = la_R*(self.R1) + (1-la_R)*(R1_update)
        self.new_R2 = la_R*(self.R2) + (1-la_R)*(R2_update)

        # loss calculation 
        cov_loss = - (torch.logdet(self.new_R1 + self.R_eps) + torch.logdet(self.new_R2 + self.R_eps)) / D
        inv_loss = self.invariance_loss(z1, z2)
        total_loss = cov_loss + self.alpha_tradeoff*inv_loss

        # This is required because new_R updated with backward.
        self.R1 = self.new_R1.detach()
        self.mu1 = self.new_mu1.detach()
        self.R2 = self.new_R2.detach()
        self.mu2 = self.new_mu2.detach()

        return {
            "cov": cov_loss,
            "inv": inv_loss,
            "total": total_loss
        }