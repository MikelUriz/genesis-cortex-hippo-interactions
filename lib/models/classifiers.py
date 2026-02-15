from typing import Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import pairwise_distances

from lib.models import EncoderDownM
from lib.models.base import BaseModel


@dataclass
class TripleClassifierConfig:
    latent_dim: int
    hidden_dim_base: int
    m: int
    n_digits: int
    n_colors: int
    image_size: int

@dataclass
class ClassifierOutput:
    digit_logits: Optional[torch.Tensor] = None
    color_logits: Optional[torch.Tensor] = None
    digit_color_pair_logits: Optional[torch.Tensor]= None


class TripleClassifier(BaseModel):
    def __init__(self, config: TripleClassifierConfig):
        super().__init__(config=config)
        self.latent_dim = config.latent_dim
        self.hidden_dim_base = config.hidden_dim_base
        self.m = config.m
        self.n_digits = config.n_digits
        self.n_colors = config.n_colors
        self.image_size = config.image_size

        digit_color_pair_mapping = np.array(np.meshgrid(
            range(self.n_digits), 
            range(self.n_colors),
        )).reshape(2,-1).T
        self.digit_color_pair_mapping = torch.tensor(digit_color_pair_mapping).long()
        
        self.image_encoder = EncoderDownM(config)

        with torch.no_grad():
            embed = self.encode(torch.randn(1,3,self.image_size,self.image_size))
            self.classification_input_size = embed.shape[-1]

        self.digit_head = nn.Linear(self.classification_input_size, self.n_digits)
        self.color_head = nn.Linear(self.classification_input_size, self.n_colors)
        self.digit_color_pair_head = nn.Linear(self.classification_input_size, int(self.n_digits*self.n_colors))


    @torch.no_grad()
    def _map_digit_color_to_pair(self, 
        digit_ids: torch.LongTensor,
        color_ids: torch.LongTensor                             
    ):
        digit_color_pair_labels = torch.vstack([digit_ids, color_ids]).T
        labels = pairwise_distances(digit_color_pair_labels, self.digit_color_pair_mapping).argmin(axis=-1)
        return torch.tensor(labels).long()


    def encode(self, pixel_values: torch.Tensor):
        h = self.image_encoder(pixel_values)
        h = h.flatten(1)
        return h
    
    
    def forward(self, pixel_values: torch.Tensor):
        h = self.encode(pixel_values)

        return ClassifierOutput(
            digit_logits=self.digit_head(h),
            color_logits=self.color_head(h),
            digit_color_pair_logits=self.digit_color_pair_head(h)
        )

    

    

