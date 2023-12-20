from typing import Any, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from solo.losses.sear import sear_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select


class SEAR(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SEAR

        Extra cfg settings:
            method_kwargs:
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                proj_output_dim (int): number of dimensions of projected features.
                alpha (float): alpha parameter for entropy calculation
                kernel_type (str): kernel type for entropy calculation (can be linear or gaussian)
                rank_calculator (str): rank calculator type for rank loss calculation
        """

        super().__init__(cfg)

        self.alpha: float = cfg.method_kwargs.alpha
        self.rank_calculator: str = cfg.method_kwargs.rank_calculator
        self.kernel_type: str = cfg.method_kwargs.kernel_type

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

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SEAR, SEAR).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")

        cfg.method_kwargs.alpha = omegaconf_select(cfg, "method_kwargs.alpha", 1.01)
        cfg.method_kwargs.kernel_type = omegaconf_select(cfg, "method_kwargs.scale_loss", "linear")
        cfg.method_kwargs.rank_calculator = omegaconf_select(cfg, "method_kwargs.rank_calculator", "frobenius")

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
        """Training step for SEAR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SEAR loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]

        sear_loss = sear_loss_func(z1, z2, kernel_type=self.kernel_type, alpha=self.alpha, rank_calculator=self.rank_calculator)

        self.log("train_sear_loss", sear_loss, on_epoch=True, sync_dist=True)
        
        return sear_loss + class_loss