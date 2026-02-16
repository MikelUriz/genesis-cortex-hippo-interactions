from typing import Optional, Dict, Tuple, Literal
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits
import numpy as np

from lib.models.base import VAEOutput, BaseModel, AEModelConfig
from lib.models.encoders import EncoderDownM
from lib.models.decoders import DecoderUpM, DecoderUpMFiLM


class CVAE(BaseModel):
    def __init__(self,
        config: AEModelConfig,
        n_labels: int, 
        n_features: int,
        n_cond_channels: int,
        image_size: int,
        reduction: Literal["mean", "sum"] = "mean"
    ):
        super().__init__(config=config)
        self.latent_dim = config.latent_dim
        self.hidden_dim_base = config.hidden_dim_base
        self.m = config.m
        self.kld_beta = config.kld_beta
        self.n_labels = n_labels
        self.n_features = n_features
        self.rec_loss = binary_cross_entropy_with_logits
        self.reduction = reduction

        #--- Image Encoder/Decoder
        self.image_encoder = EncoderDownM(config)
        self.image_decoder = DecoderUpM(config)

        with torch.no_grad():
            h = self.image_encoder(torch.randn(1,3,image_size,image_size))
            self.hidden_shape = h.shape[1:]
            self.cond_hidden_shape = (n_cond_channels, self.hidden_shape[-1], self.hidden_shape[-1])
            self.c_dim = np.prod(self.cond_hidden_shape)

        #--- Sem/Feat Embedders ---#
        self.sem_embeddings = nn.Embedding(self.n_labels, self.c_dim)
        self.feat_embeddings = nn.Embedding(self.n_features, self.c_dim)

        # bring the decoder input to have latent_dim number of channels
        self.image_decoder_input_cond_proj = nn.Conv2d(
            self.latent_dim + n_cond_channels + n_cond_channels,
            self.latent_dim,
            kernel_size=1
        )

        #--- gaussian latents proj ---#
        self.mu_proj = nn.Conv2d(self.latent_dim, self.latent_dim, 1)
        self.logvar_proj = nn.Conv2d(self.latent_dim, self.latent_dim, 1)


    def _create_conditional_image_decoder_input(self,
        hidden_states: torch.Tensor,
        sem_embedding: torch.Tensor,
        feat_embedding: torch.Tensor,
    ):
        """
        Given C=[c_sem, c_feat] embedding, concatenate it with image encoded
        latent states to make conditioning, and project everything to a 
        dimension compatible for the decoder input (latent_dim, w, h).

        hidden_states: image latent states (batch_size, latent_dim, w/2^m, h/2^m).
        sem_embedding: semantic embedding representing digit label 0-9 (batch_size, latent_dim, w/2^m, h/2^m).
        feat_embedding: feature embedding representing color label (batch_size, latent_dim, w/2^m, h/2^m).
        """
        decoder_input = torch.cat([hidden_states, sem_embedding, feat_embedding], dim=1) # (batch_size, latent_dim + c_dim + c_dim, w, h)
        decoder_input = self.image_decoder_input_cond_proj(decoder_input) # (batch_size, latent_dim, w/2^m, h/2^m)
        return decoder_input


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
        decoder_input = self._create_conditional_image_decoder_input(
            hidden_states,
            sem_embedding,
            feat_embedding
        ) # Conditional Input to Decoder by merging sem/feat embeddings with image hidden states

        reconstructed = self.image_decoder(decoder_input)
        return reconstructed
    

    def forward(self, 
        pixel_values: torch.Tensor,
        label_ids: torch.LongTensor,
        feature_ids: torch.LongTensor,
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

        if self.reduction == "mean":
            rec_loss = self.rec_loss(reconstructed, pixel_values)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec_loss + self.kld_beta*kld_loss

        elif self.reduction == "sum":
            rec_loss = self.rec_loss(reconstructed, pixel_values, reduction="none").sum(dim=(1,2,3))
            #kld_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=(1,2,3))
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec_loss + self.kld_beta*kld_loss
            loss = loss.mean()
            rec_loss = rec_loss.mean()
            #kld_loss = kld_loss.mean()

        return VAEOutput(reconstructed, loss, rec_loss, kld_loss)
    

    @torch.no_grad()
    def conditional_simulation(self,
        *,
        label_id: np.ndarray[int],
        feature_id: np.ndarray[int]
    ) -> torch.Tensor:
        """
        Simulate an RGB image [0,1] given label and feature ids.

        label_id: digit label 0-9 (MNIST).
        feature_id: color label.
        """
        device = next(iter(self.parameters())).device

        if isinstance(label_id, int):
            label_id = [label_id]
            feature_id = [feature_id]

        label_id = torch.tensor(label_id, device=device)
        feature_id = torch.tensor(feature_id, device=device)

        bs = label_id.size(0)

        z = torch.randn(bs, *self.hidden_shape, device=device)
        sem_embed = self.sem_embeddings(label_id).view(bs, *self.cond_hidden_shape)
        feat_embed = self.feat_embeddings(feature_id).view(bs, *self.cond_hidden_shape)

        reconstructed = self.decode(z, sem_embed, feat_embed) 

        if bs==1:
            return reconstructed[0]
        else:
            return reconstructed
    

class CVAEFilm(CVAE):
    def __init__(self,
        config: AEModelConfig,
        n_labels: int, 
        n_features: int,
        n_cond_channels: int,
        image_size: int,
        reduction: str = "sum" # try mean also
    ):
        super().__init__(
            config=config,
            n_labels=n_labels,
            n_features=n_features,
            n_cond_channels=n_cond_channels,
            image_size=image_size,
            reduction=reduction
        )

        config.cond_channels = int(n_cond_channels*2)
        self.image_decoder = DecoderUpMFiLM(config)


    def decode(self,
        hidden_states: torch.Tensor,
        sem_embedding: torch.Tensor,
        feat_embedding: torch.Tensor,
    ):
        cond = torch.cat([sem_embedding, feat_embedding], dim=1)
        reconstructed = self.image_decoder(hidden_states, cond)
        return reconstructed


class CVAESemEpiInputCondM(BaseModel):
    """
    Conditional VAE (RGB input image) with Input-conditioned Semantic/Episodic embeddings:
    q(z | x, c_sem, c_epi); C = [c_sem, c_epi] = f(x).
    Conditioning has a recursive logic:
    1. Input x is conditioned by C = [c_sem, c_epi],
    2. but C is first learned conditioned in x.
    3. So first x -> C, and then x_hat ~ p(z, x, C(x)).
    This cVAE is based on M-factor Down/Up-sampling Encoder/Decoder logic.
    Encoder, Decoder, and Input-conditioned Sem/Epi embedders are all generated inside.
    """
    def __init__(self,
        config: AEModelConfig,
        semantic_encoder: EncoderDownM,
        episodic_encoder: EncoderDownM
    ):
        """
        Config has the following
        latent_dim: number of channels of the compressed image. It provides the base input to the decoder.
        hidden_dim_base: the seed number of convolutional filters (doubled every m-step).
        m: upsampling steps.
        kld_beta: weight for the KL loss.

        Sem/Epi Encoders
        semantic_encoder: encoder to embed digit class labels.
        episodic_encoder: encoder to embed feature (e.g. color) ids.
        """
        super().__init__(config=config)
        self.latent_dim = config.latent_dim
        self.hidden_dim_base = config.hidden_dim_base
        self.m = config.m
        self.kld_beta = config.kld_beta
        self.rec_loss = binary_cross_entropy_with_logits

        #--- Sem/Epi Input-conditioned Encoders ---#
        self.semantic_encoder = semantic_encoder
        self.episodic_encoder = episodic_encoder

        self._sem_epi_frozen = False

        #--- Image Encoder/Decoder
        self.image_encoder = EncoderDownM(config)
        self.image_decoder = DecoderUpM(config)

        #--- input-conditioning adjustment projections ---#
        n_cond_channels = self.semantic_encoder.config.latent_dim + self.episodic_encoder.config.latent_dim

        # Bring the size of the input conditioning tensor 
        # to image input size
        self.upsampled_input_conditioning_proj = nn.Sequential(
            *[
                nn.ConvTranspose2d(
                    n_cond_channels, n_cond_channels,
                    kernel_size=3,
                    stride=2, padding=1,
                    output_padding=1
                ) for _ in range(self.m)
            ]
        )

        # Whatever the conditioning channels, bring the 
        # encoder input to 3-channels
        self.image_encoder_input_cond_proj = nn.Conv2d(
            n_cond_channels + 3, 3, 
            kernel_size=1
        )

        # Whatever the conditioning channels, bring the latent
        # decoder input to latent_dim-channels
        self.image_decoder_input_cond_proj = nn.Conv2d(
            n_cond_channels + self.latent_dim, self.latent_dim, 
            kernel_size=1
        )

        #--- gaussian latents proj ---#
        self.mu_proj = nn.Conv2d(self.latent_dim, self.latent_dim, 1)
        self.logvar_proj = nn.Conv2d(self.latent_dim, self.latent_dim, 1)


    def _create_conditional_image_encoder_input(self,
        pixel_values: torch.Tensor,
        semantic_embedding: torch.Tensor,
        episodic_embedding: torch.Tensor,
    ):
        """
        Given C=[c_sem, c_epi] embedding, upsample it to bring it to
        a dimension (Ch, W, H) compatible with the input image (3, W, H),
        than concatenate C with input image to make the conditioning, and project
        everything to a final (3, W, H) encoder input.
        """
        conditioning_embedding = torch.cat([semantic_embedding, episodic_embedding], dim=1) # (batch_size, ch_s + ch_e, w/2^m, h/2^m)
        input_conditioning = self.upsampled_input_conditioning_proj(conditioning_embedding) # (batch_size, ch_s + ch_e, w, h)
        encoder_input = torch.cat([pixel_values, input_conditioning], dim=1) # (batch_size, 3 + ch_s + ch_e, w, h)
        encoder_input = self.image_encoder_input_cond_proj(encoder_input) # (batch_size, 3, w, h)
        return encoder_input


    def _create_conditional_image_decoder_input(self,
        hidden_states: torch.Tensor,
        semantic_embedding: torch.Tensor,
        episodic_embedding: torch.Tensor,
    ):
        """
        Given C=[c_sem, c_epi] embedding, concatenate it with image encoded
        latent states to make conditioning, and project everything to a 
        dimension compatible for the decoder input (latent_dim, w, h).
        """
        conditioning_embedding = torch.cat([semantic_embedding, episodic_embedding], dim=1) # (batch_size, ch_s + ch_e, w/2^m, h/2^m)
        decoder_input = torch.cat([hidden_states, conditioning_embedding], dim=1) # (batch_size, latent_dim + ch_s + ch_e, w, h)
        decoder_input = self.image_decoder_input_cond_proj(decoder_input) # (batch_size, latent_dim, w/2^m, h/2^m)
        return decoder_input
    

    def _reparameterize(self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def encode(self, 
        pixel_values: torch.Tensor,
        semantic_embedding: torch.Tensor,
        episodic_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project input image and conditioning embeddings C to mu, logvar tensors.
        """
        encoder_input = self._create_conditional_image_encoder_input(
            pixel_values,
            semantic_embedding,
            episodic_embedding
        ) # Conditional Input to Encoder by merging sem/epi embeddings and input image
        encoder_output = self.image_encoder(encoder_input)

        mu, logvar = self.mu_proj(encoder_output), self.logvar_proj(encoder_output)
        return mu, logvar


    def decode(self,
        hidden_states: torch.Tensor,
        semantic_embedding: torch.Tensor,
        episodic_embedding: torch.Tensor,
    ):
        """
        Project hidden states and conditioning embeddings to the reconstructed image.
        """
        decoder_input = self._create_conditional_image_decoder_input(
            hidden_states,
            semantic_embedding,
            episodic_embedding
        ) # Conditional Input to Decoder by merging sem/epi embeddings and hidden states

        reconstructed = self.image_decoder(decoder_input)
        return reconstructed
    

    def get_semantic_embedding(self, 
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        return self.semantic_encoder(pixel_values)
    

    def get_episodic_embedding(self, 
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        return self.episodic_encoder(pixel_values)


    def forward(self, 
        pixel_values: torch.Tensor
    ) -> VAEOutput:
        """
        pixel_values: batch of RGB images (batch_size, 3, W, H).
        """
        semantic_embedding = self.get_semantic_embedding(pixel_values) # (batch_size, ch_s, w/2^m, h/2^m)
        episodic_embedding = self.get_episodic_embedding(pixel_values) # (batch_size, ch_e, w/2^m, h/2^m)
        
        mu, logvar = self.encode(pixel_values, semantic_embedding, episodic_embedding)
        z = self._reparameterize(mu, logvar)
        self.hidden_shape = z.shape[1:]
        reconstructed = self.decode(z, semantic_embedding, episodic_embedding)

        rec_loss = self.rec_loss(reconstructed, pixel_values)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = rec_loss + self.kld_beta*kld_loss

        return VAEOutput(reconstructed, loss, rec_loss, kld_loss)
    

    @torch.no_grad()
    def conditional_simulation(self,
        *,
        semantic_embedding: torch.Tensor,
        episodic_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Simulate an RGB image [0,1] given C=[c_sem, c_epi]. In this model, C is conditioned
        to the image input (i.e. C(x)). However, sem/epi embeddings can be generated from outside
        based on some experimental/task procedure.

        semantic_embedding: tensor of shape (ch_sem, w, h)
        episodic_embeddings: tensor of shape (ch_epi, w, h)
        """
        device = next(iter(self.parameters())).device
        ch, w, h = self.hidden_shape

        semantic_embedding = semantic_embedding.to(device)
        episodic_embedding = episodic_embedding.to(device)
        z = torch.randn(1, ch, w, h, device=device)
        reconstructed = self.decode(z, semantic_embedding[None], episodic_embedding[None])
        reconstructed = torch.clip(reconstructed, 0, 1)
        return reconstructed[0]
    

    def freeze_sem_epi(self, freeze: bool = True):
        """Enable/disable training of semantic & episodic encoders."""
        self._sem_epi_frozen = freeze
        for m in (self.semantic_encoder, self.episodic_encoder):
            for p in m.parameters():
                p.requires_grad = not freeze
        # keep BN in eval mode when frozen
        if freeze:
            self.semantic_encoder.eval()
            self.episodic_encoder.eval()
        return self


    def train(self, mode: bool = True):
        # keep usual behavior, but pin frozen encoders in eval() so BN stats don't update
        super().train(mode)
        if self._sem_epi_frozen:
            self.semantic_encoder.eval()
            self.episodic_encoder.eval()
        return self


    def save_sem_epi(self, path: str):
        data = dict(
            sem=dict(
                state_dict=self.semantic_encoder.state_dict(),
                config=self.semantic_encoder.config
            ),
            epi=dict(
                state_dict=self.episodic_encoder.state_dict(),
                config=self.episodic_encoder.config
            )
        )

        torch.save(data, path)



class CVAESemEpiInputCondMFiLM(BaseModel):
    """
    Conditional VAE (RGB input image) with Input-conditioned Semantic/Episodic embeddings and Decoder-FiLM:
    q(z | x, c_sem, c_epi); C = [c_sem, c_epi] = f(x).
    Conditioning has a recursive logic:
    1. Input x is conditioned by C = [c_sem, c_epi],
    2. but C is first learned conditioned in x.
    3. So first x -> C, and then x_hat ~ p(z, x, C(x)).
    4. Decoder -> g(z, C(x)), where g = FiLM.
    This cVAE is based on M-factor Down/Up-sampling Encoder/Decoder logic.
    Encoder, Decoder, and Input-conditioned Sem/Epi embedders are all generated inside.
    """
    def __init__(self,
        config: AEModelConfig,
        semantic_encoder: EncoderDownM,
        episodic_encoder: EncoderDownM
    ):
        """
        Config has the following
        latent_dim: number of channels of the compressed image. It provides the base input to the decoder.
        hidden_dim_base: the seed number of convolutional filters (doubled every m-step).
        m: upsampling steps.
        kld_beta: weight for the KL loss.

        Sem/Epi Encoders
        semantic_encoder: encoder to embed digit class labels.
        episodic_encoder: encoder to embed feature (e.g. color) ids.
        """
        super().__init__(config=config)
        self.latent_dim = config.latent_dim
        self.hidden_dim_base = config.hidden_dim_base
        self.m = config.m
        self.kld_beta = config.kld_beta
        self.rec_loss = binary_cross_entropy_with_logits

        #--- Sem/Epi Input-conditioned Encoders ---#
        self.semantic_encoder = semantic_encoder
        self.episodic_encoder = episodic_encoder

        self._sem_epi_frozen = False

        n_cond_channels = self.semantic_encoder.config.latent_dim + self.episodic_encoder.config.latent_dim

        #--- Image Encoder/Decoder
        self.image_encoder = EncoderDownM(config)
        config.cond_channels = n_cond_channels
        self.image_decoder = DecoderUpMFiLM(config)

        #--- gaussian latents proj ---#
        self.mu_proj = nn.Conv2d(self.latent_dim, self.latent_dim, 1)
        self.logvar_proj = nn.Conv2d(self.latent_dim, self.latent_dim, 1)


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
    

    def decode(self, z, cond):
        return self.image_decoder(z, cond)
    

    def get_semantic_embedding(self, 
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        return self.semantic_encoder(pixel_values)
    

    def get_episodic_embedding(self, 
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        return self.episodic_encoder(pixel_values)
    

    def forward(self, 
        pixel_values: torch.Tensor
    ) -> VAEOutput:
        """
        pixel_values: batch of RGB images (batch_size, 3, W, H).
        """
        semantic_embedding = self.get_semantic_embedding(pixel_values)
        episodic_embedding = self.get_episodic_embedding(pixel_values)
        cond = torch.cat([semantic_embedding, episodic_embedding], dim=1)

        mu, logvar = self.encode(pixel_values)
        z = self._reparameterize(mu, logvar)
        self.hidden_shape = z.shape[1:]
        reconstructed = self.decode(z, cond)

        rec_loss = self.rec_loss(reconstructed, pixel_values)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = rec_loss + self.kld_beta * kld_loss
        return VAEOutput(reconstructed, loss, rec_loss, kld_loss)
    

    @torch.no_grad()
    def conditional_simulation(self,
        *,
        semantic_embedding: torch.Tensor,
        episodic_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Simulate an RGB image [0,1] given C=[c_sem, c_epi]. In this model, C is conditioned
        to the image input (i.e. C(x)). However, sem/epi embeddings can be generated from outside
        based on some experimental/task procedure.

        semantic_embedding: tensor of shape (ch_sem, w, h)
        episodic_embeddings: tensor of shape (ch_epi, w, h)
        """
        device = next(iter(self.parameters())).device
        ch, w, h = self.hidden_shape

        semantic_embedding = semantic_embedding.to(device)
        episodic_embedding = episodic_embedding.to(device)
        cond = torch.cat([semantic_embedding[None], episodic_embedding[None]], dim=1)

        z = torch.randn(1, ch, w, h, device=device)
        reconstructed = self.decode(z, cond)
        reconstructed = torch.clip(reconstructed, 0, 1)
        return reconstructed[0]
    

    def freeze_sem_epi(self, freeze: bool = True):
        """Enable/disable training of semantic & episodic encoders."""
        self._sem_epi_frozen = freeze
        for m in (self.semantic_encoder, self.episodic_encoder):
            for p in m.parameters():
                p.requires_grad = not freeze
        # keep BN in eval mode when frozen
        if freeze:
            self.semantic_encoder.eval()
            self.episodic_encoder.eval()
        return self


    def train(self, mode: bool = True):
        # keep usual behavior, but pin frozen encoders in eval() so BN stats don't update
        super().train(mode)
        if self._sem_epi_frozen:
            self.semantic_encoder.eval()
            self.episodic_encoder.eval()
        return self


    def save_sem_epi(self, path: str):
        data = dict(
            sem=dict(
                state_dict=self.semantic_encoder.state_dict(),
                config=self.semantic_encoder.config
            ),
            epi=dict(
                state_dict=self.episodic_encoder.state_dict(),
                config=self.episodic_encoder.config
            )
        )

        torch.save(data, path)
        


