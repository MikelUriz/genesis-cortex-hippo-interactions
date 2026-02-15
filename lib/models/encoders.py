import torch
import torch.nn as nn
from lib.models.utils import SkipConnectionWrapper
from lib.models.base import AEModelConfig, BaseModel


class EncoderDownM(BaseModel):
    """
    Encoder for downsampling an RGB image by downsampling factor m, such that w=W/2^m, h=H/2^m.
    To ensure expected behavior, the input image must be a [3 x W x H] tensor where W,H are multiple of 2^m.
    Image size is halved m times (e.g. W,H=32, m=3, w,h=32/2^3=4, the downsampled image is [ch x 4 x 4]).
    """
    def __init__(self, config: AEModelConfig):
        """
        Config has the following
        latent_dim: number of channels of the compressed image. It provides the base input to the decoder.
        hidden_dim_base: the seed number of convolutional filters (doubled every m-step).
        m: upsampling steps.
        """
        super().__init__(config=config)
        self.latent_dim = config.latent_dim
        self.hidden_dim_base = config.hidden_dim_base
        self.m = config.m
    
        self.input_proj = nn.Sequential(
            nn.Conv2d(
                3, self.hidden_dim_base,
                kernel_size=3,
                stride=1, padding=1
            ),
            nn.BatchNorm2d(self.hidden_dim_base),
            nn.ReLU()
        )

        layers = []
        for i in range(self.m):
            Cin  = self.hidden_dim_base * (2 ** i)
            Cout = self.hidden_dim_base * (2 ** (i + 1))

            residual_block = nn.Sequential(
                nn.Conv2d(
                    Cin, Cin,
                    kernel_size=3,
                    stride=1, padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    Cin, Cin,
                    kernel_size=3,
                    stride=1, padding=1
                )
            )
            residual_block = SkipConnectionWrapper(residual_block, Cin, Cin)

            downsample_block = nn.Conv2d(
                Cin, Cout,
                kernel_size=4,
                stride=2, padding=1
            )

            layers.append(
                nn.Sequential(
                    residual_block,
                    nn.BatchNorm2d(Cin),
                    downsample_block,
                    nn.ReLU()
                )
            )

        self.layers = nn.Sequential(*layers)

        self.hidden_proj = nn.Conv2d(Cout, self.latent_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.layers(x)
        x = self.hidden_proj(x)
        return x