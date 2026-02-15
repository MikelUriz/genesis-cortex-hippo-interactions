import torch
import torch.nn as nn
from lib.models.utils import SkipConnectionWrapper, FiLM2d
from lib.models.base import AEModelConfig, BaseModel


class DecoderUpM(BaseModel):
    """
    Decoder for upsampling a compressed image [chw x h] by upsampling factor m, such that W=w*2^m, H=h*2^m.
    Compressed image size is doubled m times (e.g. w,h=4, m=3, W,H=4*2^3=32, the upsampled image is [ch x 32 x 32]).
    """
    def __init__(self, config: AEModelConfig):
        """
        Config has the following
        latent_dim: number of channels of the compressed image. It provides the base input to the decoder.
        hidden_dim_base: the seed number of convolutional filters used in the Encoder.
        m: downsampling steps.
        """
        super().__init__(config=config)
        self.latent_dim = config.latent_dim
        self.hidden_dim_base = config.hidden_dim_base
        self.m = config.m

        Cout = self.hidden_dim_base * (2**self.m)
        self.inverse_hidden_proj = nn.Sequential(
            nn.Conv2d(self.latent_dim, Cout, 1),
            nn.BatchNorm2d(Cout),
            nn.ReLU()
        )

        layers = []
        for i in range(self.m, 0, -1):
            Cin = self.hidden_dim_base * (2 ** i)
            Cout = self.hidden_dim_base * (2 ** (i - 1))

            upsample_block = nn.ConvTranspose2d(
                Cin, Cout,
                kernel_size=4,
                stride=2, padding=1,
                output_padding=0
            )

            residual_block = nn.Sequential(
                nn.Conv2d(
                    Cout, Cout,
                    kernel_size=3,
                    stride=1, padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                    Cout, Cout,
                    kernel_size=3,
                    stride=1, padding=1
                )
            )
            residual_block = SkipConnectionWrapper(residual_block, Cout, Cout)

            layers.append(
                nn.Sequential(
                    upsample_block,
                    nn.ReLU(),
                    residual_block,
                    nn.BatchNorm2d(Cout)
                )
            )

        self.layers = nn.Sequential(*layers)

        self.output_proj = nn.Conv2d(Cout, 3, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.inverse_hidden_proj(z)
        x = self.layers(x)
        x = self.output_proj(x)
        return x
    

class DecoderUpMFiLM(BaseModel):
    """
    Same upsampling topology as DecoderUpM, but inject FiLM after BN at each stage.
    Forward now expects: forward(z, cond).
    """
    def __init__(self, config: AEModelConfig):
        super().__init__(config=config)
        self.latent_dim = config.latent_dim
        self.hidden_dim_base = config.hidden_dim_base
        self.m = config.m
        self.cond_channels = config.cond_channels

        Cout = self.hidden_dim_base * (2**self.m)
        self.inverse_hidden_proj = nn.Sequential(
            nn.Conv2d(self.latent_dim, Cout, 1),
            nn.BatchNorm2d(Cout),
            nn.ReLU()
        )

        blocks = []
        for i in range(self.m, 0, -1):
            Cin = self.hidden_dim_base * (2 ** i)
            Cout = self.hidden_dim_base * (2 ** (i - 1))

            upsample_block = nn.ConvTranspose2d(Cin, Cout, kernel_size=4, stride=2, padding=1)

            residual_block = nn.Sequential(
                nn.Conv2d(Cout, Cout, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(Cout, Cout, kernel_size=3, stride=1, padding=1)
            )
            residual_block = SkipConnectionWrapper(residual_block, Cout, Cout)

            blocks.append(nn.ModuleDict({
                "up": upsample_block,
                "act": nn.ReLU(),
                "residual": residual_block,
                "bn": nn.BatchNorm2d(Cout),
                "film": FiLM2d(self.cond_channels, Cout)
            }))

        self.blocks = nn.ModuleList(blocks)
        self.output_proj = nn.Conv2d(Cout, 3, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, cond: torch.Tensor, verbose: int = 0) -> torch.Tensor:
        feature_map = None

        x = self.inverse_hidden_proj(z)

        for i, blk in enumerate(self.blocks):
            x = blk["up"](x)
            x = blk["act"](x)
            x = blk["residual"](x)
            x = blk["bn"](x)
            x = blk["film"](x, cond)

            if i == len(self.blocks)-verbose:
                feature_map = x

        x = self.output_proj(x)

        if verbose != 0:
            return x, feature_map

        return x