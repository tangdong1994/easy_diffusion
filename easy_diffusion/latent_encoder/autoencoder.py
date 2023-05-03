import torch
from torch import nn
import numpy as np
from easy_diffusion.model_block import Encoder, Decoder, DiagonalGaussianDistribution

class SimpleAutoEncoder(nn.Module):
    """A simple auto encoder architecture."""
    def __init__(self, in_channel=1, channels=[4, 8, 32],):
        """
        Args:
            in_channel: The number of channel for input image
            channels: The number of channels for feature maps of each resolution.
        """
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channel, channels[0], 3, stride=1, bias=True),
                       nn.BatchNorm2d(channels[0]),
                       nn.SiLU(),
                       nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=True),
                       nn.BatchNorm2d(channels[1]),
                       nn.SiLU(),
                       nn.Conv2d(channels[1], channels[2], 3, stride=1, bias=True),
                       nn.BatchNorm2d(channels[2]),
                       )
        self.decoder = nn.Sequential(nn.ConvTranspose2d(channels[2], channels[1], 3, stride=1, bias=True),
                       nn.BatchNorm2d(channels[1]),
                       nn.SiLU(),
                       nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=True, output_padding=1),
                       nn.BatchNorm2d(channels[0]),
                       nn.SiLU(),
                       nn.ConvTranspose2d(channels[0], in_channel, 3, stride=1, bias=True),
                       nn.Sigmoid())
    def forward(self, x):
        output = self.decoder(self.encoder(x))
        return output
    
    
class AutoEncoderKL(nn.Module):
    def __init__(self, 
                 encoder_cfg: dict,
                 decoder_cfg: dict,
                 embed_dim: int
                ):
        super().__init__()
        self.encoder = Encoder(**encoder_cfg)
        self.decoder = Decoder(**decoder_cfg)
        self.embed_dim = embed_dim
        self.quant_conv = torch.nn.Conv2d(2*encoder_cfg["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder_cfg["z_channels"], 1)
        
    def encode(self, x):
        x = self.encoder(x)
        moments = self.quant_conv(x)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    
    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def forward(self, x, sample_posterior=True):
        x = self.encode(x)
        if sample_posterior:
            z = x.sample()
        else:
            z = x.mode()
        dec = self.decode(z)
        return dec, x
        