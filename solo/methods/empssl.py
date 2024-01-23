from typing import Any, List, Sequence, Dict

import omegaconf
import torch
import torch.nn as nn
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
import torch.nn.functional as F
from solo.losses.empssl import cosine_similarity_loss_func, calculate_TCR_for_list
from solo.utils.metrics import accuracy_at_k

class EMPSSL(BaseMethod):
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
        self.tcr_epsilon = cfg.tcr_epsilon
        self.inv_weight = cfg.inv_weight
        self.tcr_weight = cfg.tcr_weight

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

        cfg = super(EMPSSL, EMPSSL).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")

        cfg.num_patches = omegaconf_select(cfg, "augmentations.0.num_crops", 100)
        cfg.tcr_epsilon = omegaconf_select(cfg, "method_kwargs.tcr_epsilon", 0.2)
        cfg.tcr_weight = omegaconf_select(cfg, "method_kwargs.tcr_weight", 1)
        cfg.inv_weight = omegaconf_select(cfg, "method_kwargs.inv_weight", 200)

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: List[torch.Tensor]) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        # concat all patches together for quick forward pass
        X = torch.cat(X, dim=0)
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)

        # forward pass
        feats = self.backbone(X)
        logits = self.classifier(self.chunk_avg(feats, self.num_patches))
        z = self.projector(feats)

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
        outs= self(X)

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
        inv_loss = cosine_similarity_loss_func(z_list, z_avg)
        tcr_loss = calculate_TCR_for_list(z_list, self.tcr_epsilon)
        loss = (self.inv_weight*inv_loss) + (self.tcr_weight*tcr_loss)

        return loss + outs["class_loss"]

    def validation_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = None,
        update_validation_step_outputs: bool = True,
    ) -> Dict[str, Any]:
        _, X, targets = batch

        batch_size = targets.size(0)
        outs= self(X)
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