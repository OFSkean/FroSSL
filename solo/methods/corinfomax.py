from typing import Any, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from solo.losses import CorInfoMax_Loss
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select


class CorInfoMax(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements CorInfoMax.

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                R_ini: 1.0          # coefficient of initial covariance
                la_R: 1e-2         # forgetting factor for covariance
                la_mu: 1e-2        # forgetting factor for mean
                R_eps_weight: 1e-6 # diagonal perturbation factor of covariance matrix R1
        """

        super().__init__(cfg)

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.loss = CorInfoMax_Loss(output_dim=proj_output_dim, 
                                    R_ini=cfg.method_kwargs.R_ini, 
                                    la_R=cfg.method_kwargs.la_R, 
                                    la_mu=cfg.method_kwargs.la_mu, 
                                    R_eps_weight=cfg.method_kwargs.R_eps_weight,
                                    alpha_tradeoff=cfg.method_kwargs.alpha_tradeoff)



    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(CorInfoMax, CorInfoMax).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")

        cfg.method_kwargs.R_ini = omegaconf_select(cfg, "method_kwargs.R_ini", 1.0)
        cfg.method_kwargs.la_R = omegaconf_select(cfg, "method_kwargs.la_R", 1e-2)
        cfg.method_kwargs.la_mu = omegaconf_select(cfg, "method_kwargs.la_mu", 1e-2)
        cfg.method_kwargs.R_eps_weight = omegaconf_select(cfg, "method_kwargs.R_eps_weight", 1e-6)
        cfg.method_kwargs.alpha_tradeoff = omegaconf_select(cfg, "method_kwargs.alpha_tradeoff", 500)

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X):
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for CorInfoMax reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of CorInfoMax loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]

        loss_vals = self.loss(z1, z2)

        self.log("train_cov_loss", loss_vals["cov"], sync_dist=True)
        self.log("train_inv_loss", loss_vals["inv"],  sync_dist=True)
        self.log("train_total_loss", loss_vals["total"], sync_dist=True)
        
        return loss_vals["total"] + class_loss