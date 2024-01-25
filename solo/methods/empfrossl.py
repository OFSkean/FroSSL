from typing import Any, List, Sequence, Dict

import omegaconf
import torch
import torch.nn as nn
from solo.losses.searmse import mutliview_searmse_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
import torch.nn.functional as F
from solo.utils.metrics import accuracy_at_k

class EMPFROSSL(BaseMethod):
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

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        self.num_patches = cfg.num_patches
        self.num_patches_val = cfg.num_patches_val
        self.alpha: float = cfg.method_kwargs.alpha
        self.kernel_type: str = cfg.method_kwargs.kernel_type
        self.entropy_cutoff: float = cfg.method_kwargs.entropy_cutoff
        self.cutoff_type: str = cfg.method_kwargs.cutoff_type


        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
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

        cfg = super(EMPFROSSL, EMPFROSSL).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")

        cfg.num_patches = omegaconf_select(cfg, "augmentations.0.num_crops", 20)
        cfg.num_patches_val = omegaconf_select(cfg, "method_kwargs.num_crops_val", 20)
        cfg.method_kwargs.alpha = omegaconf_select(cfg, "method_kwargs.alpha", 2)
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

    def forward(self, X: List[torch.Tensor], n_patches: int) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        # concat all patches together for quick forward pass
        X = torch.cat(X, dim=0)

        # forward pass
        feats = self.backbone(X)
        z = F.normalize(self.projector(feats), p=2)
        logits = self.classifier(self.chunk_avg(feats, n_patches))

        return {"logits": logits, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SEARMSE reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SEARMSE loss and classification loss.
        """

        # DUE TO THE PATCH NATURE OF THIS METHOD, WE NEED TO MODIFY THE TRAINING STEP
        # PRETTY EXTENSIVELY FROM WHAT IS USED ELSEWHERE IN SOLO-LEARN
        
        # forward pass
        _, X, targets = batch
        outs= self(X, self.num_patches)

        # unchunk projections
        z_list = outs["z"].chunk(self.num_patches, dim=0)
        z_avg = self.chunk_avg(outs["z"], self.num_patches)

        # online classifier loss and stats
        outs = self.get_online_classifier_loss(outs, outs["logits"], targets)
        metrics = {
            "train_class_loss": outs["class_loss"],
            "train_acc1": outs["acc1"],
            "train_acc5": outs["acc5"],
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        
        # calculate loss
        frossl_loss = mutliview_searmse_loss_func(z_list, entropy_weight=self.entropy_cutoff)

        self.log("train_frossl_loss", frossl_loss)

        return frossl_loss + outs["class_loss"]

    def validation_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = None,
        update_validation_step_outputs: bool = True,
    ) -> Dict[str, Any]:
        _, X, targets = batch

        batch_size = targets.size(0)
        outs= self(X, self.num_patches_val)
        outs = self.get_online_classifier_loss(outs, outs["logits"], targets)

        metrics = {
            "batch_size": batch_size,
            "val_loss": outs["class_loss"],
            "val_acc1": outs["acc1"],
            "val_acc5": outs["acc5"],
        }
        if update_validation_step_outputs:
            self.validation_step_outputs.append(metrics)
        return metrics
    

    def get_online_classifier_loss(self, outs, logits, targets):
        top_k_max = min(5, logits.size(1))
        outs["acc1"], outs["acc5"] = accuracy_at_k(logits, targets, top_k=(1, top_k_max))
        outs["class_loss"] = F.cross_entropy(logits, targets, ignore_index=-1)

        return outs
    
    def chunk_avg(self, x,n_chunks=2,normalize=False):
        x_list = x.chunk(n_chunks,dim=0)
        x = torch.stack(x_list,dim=0)
        if not normalize:
            return x.mean(0)
        else:
            return F.normalize(x.mean(0),dim=1)