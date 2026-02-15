import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits

from lib.models.base import VAEOutput, BaseModel, AEModelConfig
from lib.models.encoders import EncoderDownM
from lib.models.decoders import DecoderUpMFiLM


@dataclass
class ConfigBetaVAEForEmbedding:
    input_dim: int
    gamma: int = 1000
    C_start: int = 0
    C_max: int = 25
    C_iters_to_max: int = 50000


class BetaVAEForEmbedding(BaseModel):
    def __init__(self, config: ConfigBetaVAEForEmbedding):
        super().__init__(config=config)
        self.global_step = 0
        self.C_start = self.config.C_start
        self.C_max = self.config.C_max
        self.C_iters_to_max = self.config.C_iters_to_max
        self.rec_loss = mse_loss

        #--- Encoder/Decoder ---#
        self.encoder = nn.Sequential(
            nn.Linear(self.config.input_dim, 512), nn.LayerNorm(512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 512), nn.ReLU(inplace=True),
            nn.Linear(512, self.config.input_dim)
        )

        #--- gaussian latents proj ---#
        self.mu_proj = nn.Linear(256, 64)
        self.logvar_proj = nn.Linear(256, 64)

    def _capacity_schedule(self):
        if self.C_iters_to_max <= 0:
            return torch.tensor(self.C_max)
        t = min(self.global_step / self.C_iters_to_max, 1.)
        return torch.tensor(self.C_start + t*(self.C_max - self.C_start))

    def _reparameterize(self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu, logvar = self.mu_proj(h), self.logvar_proj(h)
        return mu, logvar
    
    def decode(self, h: torch.Tensor):
        return self.decoder(h)

    def forward(self, x) -> VAEOutput:
        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        reconstructed = self.decode(z)

        C_t = self._capacity_schedule()
        if self.training:
            self.global_step += 1

        rec_loss = self.rec_loss(reconstructed, x, reduction="none").sum(dim=-1)
        kld_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        loss = rec_loss + self.config.gamma*torch.abs(kld_loss - C_t.to(kld_loss.device))
        loss = loss.mean()
        rec_loss = rec_loss.mean()
        kld_loss = kld_loss.mean()

        return VAEOutput(reconstructed, loss, rec_loss, kld_loss)
    

###


@dataclass
class ConfigBetaVAEForImages:
    latent_dim: int
    hidden_dim_base: int
    m: int
    image_size: int
    n_labels: int
    n_features: int
    n_cond_channels: int
    gamma: int = 1000
    C_start: int = 0
    C_max: int = 25
    C_iters_to_max: int = 50000


class BetaVAEForImages(BaseModel):
    def __init__(self, config: ConfigBetaVAEForImages):
        super().__init__(config=config)
        self.global_step = 0
        self.C_start = self.config.C_start
        self.C_max = self.config.C_max
        self.C_iters_to_max = self.config.C_iters_to_max
        self.rec_loss = binary_cross_entropy_with_logits

        #--- Image Encoder/Decoder
        self.image_encoder = EncoderDownM(AEModelConfig(
            latent_dim=self.config.latent_dim,
            hidden_dim_base=self.config.hidden_dim_base,
            m=self.config.m
        ))
        self.image_decoder = DecoderUpMFiLM(AEModelConfig(
            latent_dim=self.config.latent_dim,
            hidden_dim_base=self.config.hidden_dim_base,
            m=self.config.m,
            cond_channels=int(self.config.n_cond_channels*2)
        ))

        with torch.no_grad():
            h = self.image_encoder(torch.randn(1, 3, self.config.image_size, self.config.image_size))
            self.hidden_shape = h.shape[1:]
            self.cond_hidden_shape = (self.config.n_cond_channels, self.hidden_shape[-1], self.hidden_shape[-1])
            self.c_dim = np.prod(self.cond_hidden_shape)

        #--- Sem/Feat Embedders ---#
        self.sem_embeddings = nn.Embedding(self.config.n_labels, self.c_dim)
        self.feat_embeddings = nn.Embedding(self.config.n_features, self.c_dim)

        # bring the decoder input to have latent_dim number of channels
        self.image_decoder_input_cond_proj = nn.Conv2d(
            self.config.latent_dim + self.config.n_cond_channels + self.config.n_cond_channels,
            self.config.latent_dim,
            kernel_size=1
        )

        #--- gaussian latents proj ---#
        self.mu_proj = nn.Conv2d(self.config.latent_dim, self.config.latent_dim, 1)
        self.logvar_proj = nn.Conv2d(self.config.latent_dim, self.config.latent_dim, 1)

    def _capacity_schedule(self):
        if self.C_iters_to_max <= 0:
            return torch.tensor(self.C_max)
        t = min(self.global_step / self.C_iters_to_max, 1.)
        return torch.tensor(self.C_start + t*(self.C_max - self.C_start))

    def _reparameterize(self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, pixel_values):
        h = self.image_encoder(pixel_values)
        mu, logvar = self.mu_proj(h), self.logvar_proj(h)
        return mu, logvar
    
    def decode(self,
        hidden_states: torch.Tensor,
        sem_embedding: torch.Tensor,
        feat_embedding: torch.Tensor,
    ):
        """
        Project hidden states and conditioning embeddings to the reconstructed image.

        hidden_states: image latent states (batch_size, latent_dim, w/2^m, h/2^m).
        sem_embedding: semantic embedding representing digit label 0-9 (batch_size, latent_dim, w/2^m, h/2^m).
        feat_embedding: feature embedding representing color label (batch_size, latent_dim, w/2^m, h/2^m).
        """
        cond = torch.cat([sem_embedding, feat_embedding], dim=1)
        reconstructed = self.image_decoder(hidden_states, cond)
        return reconstructed

    def forward(self, 
        pixel_values: torch.Tensor,
        label_ids: torch.LongTensor,
        feature_ids: torch.LongTensor
    ) -> VAEOutput:
        """
        pixel_values: batch of RGB images (batch_size, 3, W, H).
        label_ids: digit label 0-9 (MNIST).
        feature_ids: color label.
        """

        sem_embed = self.sem_embeddings(label_ids).view(-1, *self.cond_hidden_shape)
        feat_embed = self.feat_embeddings(feature_ids).view(-1, *self.cond_hidden_shape)
        
        mu, logvar = self.encode(pixel_values)
        z = self._reparameterize(mu, logvar)
        reconstructed = self.decode(z, sem_embed, feat_embed)

        C_t = self._capacity_schedule()
        if self.training:
            self.global_step += 1

        rec_loss = self.rec_loss(reconstructed, pixel_values, reduction="none").sum(dim=(1,2,3))
        kld_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=(1,2,3))
        loss = rec_loss + self.config.gamma*torch.abs(kld_loss - C_t.to(kld_loss.device))
        loss = loss.mean()
        rec_loss = rec_loss.mean()
        kld_loss = kld_loss.mean()

        return VAEOutput(reconstructed, loss, rec_loss, kld_loss)