import torch
from dataclasses import dataclass


@dataclass
class Color:
    RED=[1,0,0]
    GREEN=[0,1,0]
    BLUE=[0,0,1]
    CYAN=[0,1,1]
    YELLOW=[1,1,0]
    MAGENTA=[1,0,1]


def colorize(x: torch.Tensor, rgb_code: Color) -> torch.Tensor:
    """
    x: tensor of shape (H, W) representing a 1-Ch normalized (in [0,1]) image
    rgb_code: RGB unit vector indicating color [RED, GREEN, BLUE, YELLOW]
    """
    rgb_mask = torch.tensor(rgb_code, dtype=x.dtype, device=x.device).view(3, 1, 1)
    x_colored = x.repeat(3,1,1) * rgb_mask 
    return x_colored