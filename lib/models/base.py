from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class AEModelConfig:
    latent_dim: int
    hidden_dim_base: int
    m: int
    kld_beta: Optional[float] = None # only for VAE
    cond_channels: Optional[int] = None # only for FiLM


@dataclass
class VAEOutput:
    rec: torch.Tensor
    loss: torch.Tensor
    rec_loss: torch.Tensor
    kld_loss: torch.Tensor

    def __post_init__(self):
        self.rec_loss = self.rec_loss.detach()
        self.kld_loss = self.kld_loss.detach()


class BaseModel(nn.Module):
    def __init__(self, config: AEModelConfig):
        super().__init__()
        self.config = config

    def show_number_of_parameters(self):
        tot_params = sum(p.numel() for p in self.parameters())
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        tot_params = round(tot_params / 1e+6, 3)
        train_params = round(train_params / 1e+6, 3)
        print(f"{train_params}M / {tot_params}M")