from typing import Any, List, Sequence, Dict

import omegaconf
import torch
import torch.nn as nn
from solo.losses.searmse import searmse_loss_func, mutliview_searmse_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
import torch.nn.functional as F

class SEARMSE(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SEAR

        Extra cfg settings:
            method_kwargs:
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                proj_output_dim (int): number of dimensions of projected features.
                alpha (float): alpha parameter for entropy calculation
                kernel_type (str): kernel type for entropy calculation (can be linear or gaussian)
        """

        super().__init__(cfg)

        self.alpha: float = cfg.method_kwargs.alpha
        self.kernel_type: str = cfg.method_kwargs.kernel_type
        self.entropy_cutoff: float = cfg.method_kwargs.entropy_cutoff
        self.cutoff_type: str = cfg.method_kwargs.cutoff_type

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

        cfg = super(SEARMSE, SEARMSE).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")

        cfg.method_kwargs.alpha = omegaconf_select(cfg, "method_kwargs.alpha", 1.01)
        cfg.method_kwargs.kernel_type = omegaconf_select(cfg, "method_kwargs.scale_loss", "linear")
        cfg.method_kwargs.entropy_cutoff = omegaconf_select(cfg, "method_kwargs.entropy_cutoff", 0.2)
        cfg.method_kwargs.cutoff_type = omegaconf_select(cfg, "method_kwargs.cutoff_type", "linear")
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
        """Training step for SEARMSE reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SEARMSE loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z = out["z"]

        if self.cutoff_type == "linear":
            entropy_weight = max(self.entropy_cutoff, 1 - (self.trainer.current_epoch / self.trainer.max_epochs))
        elif self.cutoff_type == "constant":
            entropy_weight = self.entropy_cutoff
        else:
            raise NotImplementedError(f"{self.cutoff_type} not implemented")
        
        self.log("entropy_weight", entropy_weight, on_epoch=True, sync_dist=True)

        if len(z) == 2:
            z1 = z[0]
            z2 = z[1]
            searmse_loss = searmse_loss_func(z1, z2, kernel_type=self.kernel_type, alpha=self.alpha, entropy_weight=entropy_weight, logger=self.log)
        else:
            searmse_loss = mutliview_searmse_loss_func(z, entropy_weight=entropy_weight, logger=self.log)
            
        # self.log("train_searmse_loss", searmse_loss, on_epoch=True, sync_dist=True)
        
        return searmse_loss + class_loss