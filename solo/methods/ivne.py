from typing import Any, List, Sequence, Dict

import omegaconf
import torch
import torch.nn as nn
from solo.losses.ivne import ivne_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
import torch.nn.functional as F

class IVNE(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements I-VNE

        Extra cfg settings:
            method_kwargs:
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                proj_output_dim (int): number of dimensions of projected features.
                alpha_vne (float): weight of the von neumann entropy loss.
                alpha_cos (float): weight of the cosine similarity loss.
                
        """

        super().__init__(cfg)
        self.alpha_vne = cfg.method_kwargs.alpha_vne
        self.alpha_cos = cfg.method_kwargs.alpha_cos

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        if cfg.force_same_projector_size:
            print("Forcing same projector size!!!!!")
            proj_output_dim = proj_hidden_dim

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

        cfg = super(IVNE, IVNE).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        
        # defaults from https://github.com/jaeill/CVPR23-VNE/blob/c63961b05b66bb1ddeadf5ff129c7f1aabda392f/examples/i-vne%2B/train_ivne_imagenet100.py#L29
        cfg.method_kwargs.alpha_inv = omegaconf_select(cfg, "method_kwargs.alpha_vne", 1.0)
        cfg.method_kwargs.alpha_mse = omegaconf_select(cfg, "method_kwargs.alpha_cos", 4.16)

        cfg.force_same_projector_size = omegaconf_select(cfg, "method_kwargs.force_same_projector_size", False)
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
        """Training step for FroSSL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of FroSSL loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z = out["z"]

        z1 = z[0]
        z2 = z[1]
        ivne_loss = ivne_loss_func(z1, z2, self.alpha_cos, self.alpha_vne)

            
        self.log("train_ivne_loss", ivne_loss, on_epoch=True, sync_dist=True)
        
        return ivne_loss + class_loss