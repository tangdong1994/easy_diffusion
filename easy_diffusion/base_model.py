import torch
from torch import nn
from easy_diffusion.model_block import SpatialTransformer, PSPModule, ResBlock,ResBlock2,Downsample,TimestepBlock,TimestepEmbedSequential,Upsample,ResBlock2
import functools
import numpy as np
from einops import rearrange
import math

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights (frequencies) during initialization. 
        # These weights (frequencies) are fixed during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        # Cosine(2 pi freq x), Sine(2 pi freq x)
        x_proj = x[:, None] * self.W[None, :].to(x.device) * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps.
    Allow time repr to input additively from the side of a convolution layer.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None] 
        # this broadcast the 2d tensor to 4d, add the same value across space. 
        

#@title Diffusion constant and noise strength
device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

      Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  

      Returns:
        The standard deviation.
    """    
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

      Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

      Returns:
        The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)
  
sigma =  25.0#@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


#@title Defining a time-dependent score-based model (double click to expand or collapse)

class UNet(nn.Module):
    """score-based model built upon U-Net architecture. 这里的skip connection采用的是concat的方式"""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.time_embed = nn.Sequential(
              GaussianFourierProjection(embed_dim=embed_dim),
              nn.Linear(embed_dim, embed_dim)
              )
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    
        ########### YOUR CODE HERE (3 lines)
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        #########################
    
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])   
     

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
  
    def forward(self, x, t, y=None): 
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.time_embed(t))    
        # Encoding path
        h1 = self.conv1(x)  + self.dense1(embed)   
        ## Incorporate information from t
        ## Group normalization
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        ########## YOUR CODE HERE (2 lines)
        # h3 = ...    # conv, dense
        #   apply activation function
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        ############
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
    
#@title Alternative time-dependent score-based model (double click to expand or collapse)

class UNet_res(nn.Module):
    """ score-based model built upon U-Net architecture. 这里skip connection采用的是add方式"""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.time_embed = nn.Sequential(
              GaussianFourierProjection(embed_dim=embed_dim),
              nn.Linear(embed_dim, embed_dim)
              )
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)     #  + channels[2]
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)     #  + channels[1]
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1) #  + channels[0]

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, y=None): 
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.time_embed(t))    
        # Encoding path
        h1 = self.conv1(x)  + self.dense1(embed)   
        ## Incorporate information from t
        ## Group normalization
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(h + h3)
        h += self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        h = self.tconv2(h + h2)
        h += self.dense7(embed)
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(h + h1)

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
    

class UNet_Tranformer(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256, 
                   text_dim=256, nClass=10, num_heads=1):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings of time.
          text_dim:  the embedding dimension of text / digits. 
          nClass:    number of classes you want to model.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
            )
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.attn3 = SpatialTransformer(channels[2], text_dim, num_heads=num_heads) 

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    
        # YOUR CODE: interleave some attention layers with conv layers
        self.attn4 = SpatialTransformer(channels[3], text_dim, num_heads=num_heads)                        ######################################

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])   

        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)     #  + channels[2]
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)     #  + channels[1]
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0], 1, 3, stride=1) #  + channels[0]

        # The swish activation function
        self.act = nn.SiLU() # lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        if text_dim is not None:
            self.cond_embed = nn.Embedding(nClass, text_dim)
  
    def forward(self, x, t, y=None): 
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.time_embed(t)) 
        if y is not None:
            y_embed = self.cond_embed(y).unsqueeze(1)
        else:
            y_embed = y
        # print("y_embed:",y_embed.shape,y.shape)
        # Encoding path
        h1 = self.conv1(x) + self.dense1(embed) 
        ## Incorporate information from t
        ## Group normalization
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        h3 = self.attn3(h3, y_embed) # Use your attention layers
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))
        # Your code: Use your additional attention layers! 
        h4 = self.attn4(h4, y_embed)       ##################### ATTENTION LAYER COULD GO HERE IF ATTN4 IS DEFINED

        # Decoding path
        h = self.tconv4(h4) + self.dense5(embed)
        ## Skip connection from the encoding path
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(h + h3) + self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        h = self.tconv2(h + h2) + self.dense7(embed)
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(h + h1)

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
    
class UNet_Tranformer_Arch(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, input_channels=1, channels=[32, 64, 128, 256], embed_dim=256, 
                   text_dim=256, nClass=10, num_heads=1, image_size=256, psp_ratio=2, use_psp=False):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings of time.
          text_dim:  the embedding dimension of text / digits. 
          nClass:    number of classes you want to model.
        """
        super().__init__()
        self.use_psp = use_psp
        # Gaussian random feature embedding layer for time
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
            )
        # Encoding layers where the resolution decreases
        if use_psp:
            pool_sizes = [int(image_size/(2**i)) for i in range(1,4)]
            self.psp1 = PSPModule(channels[0], psp_ratio*channels[0], pool_sizes=pool_sizes)
            self.psp2 = PSPModule(channels[1], psp_ratio*channels[1], pool_sizes=pool_sizes)
            self.psp3 = PSPModule(channels[2], psp_ratio*channels[2], pool_sizes=pool_sizes)
            self.psp4 = PSPModule(channels[3], psp_ratio*channels[3], pool_sizes=pool_sizes)
            self.tpsp4 = PSPModule(channels[2], psp_ratio*channels[2], pool_sizes=pool_sizes)
            self.tpsp3 = PSPModule(channels[1], psp_ratio*channels[1], pool_sizes=pool_sizes)
            self.tpsp2 = PSPModule(channels[0], psp_ratio*channels[0], pool_sizes=pool_sizes)
            # self.tpsp1 = PSPModule(channels[0], 2*channels[0])
        self.conv1 = nn.Conv2d(input_channels, channels[0], 3, stride=1,padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.attn3 = SpatialTransformer(channels[2], text_dim, num_heads=num_heads) 

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    
        # YOUR CODE: interleave some attention layers with conv layers
        self.attn4 = SpatialTransformer(channels[3], text_dim, num_heads=num_heads)                        ######################################

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, padding=1,output_padding=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])   

        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, padding=1,bias=False, output_padding=1)     #  + channels[2]
#         self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False) 
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, padding=1,bias=False, output_padding=1)     #  + channels[1]
#         self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.Conv2d(channels[0], input_channels, 3, stride=1,padding=1) #  + channels[0]

        # The swish activation function
        self.act = nn.SiLU() # lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        if text_dim is not None:
            self.cond_embed = nn.Embedding(nClass, text_dim)
  
    def forward(self, x, t, y=None): 
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.time_embed(t)) 
        if y is not None:
            y_embed = self.cond_embed(y).unsqueeze(1)
        else:
            y_embed = y
        # print("y_embed:",y_embed.shape,y.shape)
        # Encoding path
        h1 = self.conv1(x) + self.dense1(embed) 
        ## Incorporate information from t
        ## Group normalization
        h1 = self.act(self.gnorm1(h1))
        if self.use_psp:
            h1 = self.psp1(h1)
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        if self.use_psp:
            h2 = self.psp2(h2)
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        if self.use_psp:
            h3 = self.psp3(h3)
        h3 = self.attn3(h3, y_embed) # Use your attention layers
#         print("h3",h3.shape)
        h4 = self.conv4(h3) + self.dense4(embed)
#         print("h4",h4.shape)
        h4 = self.act(self.gnorm4(h4))
        if self.use_psp:
            h4 = self.psp4(h4)
#         print("h4",h4.shape)
        # Your code: Use your additional attention layers! 
        h4 = self.attn4(h4, y_embed)       ##################### ATTENTION LAYER COULD GO HERE IF ATTN4 IS DEFINED

#         print("h4",h4.shape)
        # Decoding path
        h = self.tconv4(h4) + self.dense5(embed)
        ## Skip connection from the encoding path
        h = self.act(self.tgnorm4(h))
#         print("h",h.shape)
        if self.use_psp:
            h = self.tpsp4(h)
#         print("h",h.shape)
#         print(embed.shape)
#         print("h3",h3.shape)
        h = self.tconv3(h + h3) + self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        if self.use_psp:
            h = self.tpsp3(h)
#         print(h.shape)
#         print(h2.shape)
        h = self.tconv2(h + h2) + self.dense7(embed)
        h = self.act(self.tgnorm2(h))
#         print(h.shape)
        if self.use_psp:
            h = self.tpsp2(h)
        h = self.tconv1(h + h1)
#         print(h.shape)
#         print(h1.shape)
        

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
    
class UNetTransformerArchResBlock(nn.Module):
    def __init__(self, marginal_prob_std, in_channels=1, model_channels=32, channels=[32, 64, 128, 256], embed_dim=256, res_num=2,dropout=0,context_dim=512,
                   text_dim=512, context_embeding=512,resblock_updown=False, num_heads=1, image_size=256, psp_ratio=2, use_psp=False,dims=2,conv_resample=True):
        super().__init__()
#         self.time_embed = nn.Sequential(
#             GaussianFourierProjection(embed_dim=embed_dim),
            
#             nn.Linear(embed_dim, embed_dim)
#         )
#         self.time = get_timestep_embedding(,embed_dim)
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(embed_dim,
                            embed_dim),
            torch.nn.Linear(embed_dim,
                            embed_dim),
        ])
        self.embed_dim = embed_dim
        # n,in_channels,h,w --> n,model_channels,h,w
        self.first_block = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
        ])
        ch = model_channels
        input_block_chans = [model_channels]
        for idx, up_channel in enumerate(channels):
            for res_i in range(res_num):
                layers = [ResBlock(
                            ch,
                            embed_dim,
                            dropout,
                            out_channels=up_channel,
                            dims=dims,
                    )]
                ch = up_channel
                input_block_chans.append(ch)
                if idx!=0:
                # if res_i == 0 and idx%2==0:
                    layers.append(SpatialTransformer(ch, context_dim, num_heads=num_heads))
                self.first_block.append(TimestepEmbedSequential(*layers))
            
            out_ch = ch
            self.first_block.append(TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            ways="down",
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )))
            input_block_chans.append(ch)
            
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch,embed_dim, dropout,dims=dims),
            SpatialTransformer(ch, context_dim, num_heads=num_heads),
            ResBlock(ch,embed_dim,dropout,dims=dims))
        
        self.last_block = nn.ModuleList([])
        for idx, up_channel in list(enumerate(channels))[::-1]:
            for i in range(res_num+1):
                ich = input_block_chans.pop()
                layers = [ResBlock(
                            ch+ich,
                            embed_dim,
                            dropout,
                            out_channels=up_channel,
                            dims=dims,
                    )]
                ch = up_channel
                if idx!=0:
                # if i == 0 and idx%2==0:
                    layers.append(
                        SpatialTransformer(ch, context_dim, num_heads=num_heads)
                    )
                out_ch = ch
                if i == 0:
                    layers.append(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                ways="up",
                            )
                            if resblock_updown
                            else Upsample(
                                ch, conv_resample, dims=dims, out_channels=out_ch
                            )

                        )
                self.last_block.append(TimestepEmbedSequential(*layers))
        self.out = nn.Sequential(
            nn.GroupNorm(32, num_channels=ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1, bias=False),
        )
        self.last = nn.Tanh()
        # The swish activation function
        self.act = nn.SiLU() # lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        if context_dim is not None:
            self.cond_embed = nn.Linear(context_embeding, context_dim)
  
    def forward(self, x, t, y=None): 
        # Obtain the Gaussian random feature embedding for t   
        #emb = self.time_embed(t)
        temb = get_timestep_embedding(t, self.embed_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        emb = self.temb.dense[1](temb)
        hs = []
        if y is not None:
#             if len(y.shape)<2:
#                 y = y.unsqueeze(-1)
            context = self.cond_embed(y)#.unsqueeze(1)
        else:
            context = y
        h = x#x.type(self.dtype)
        for module in self.first_block:
            #print("dd",module)
            h = module(h, emb, context)
            hs.append(h)
            # print(h.shape)
        h = self.middle_block(h, emb, context)
        for module in self.last_block:
            hs_pop = hs.pop()
            # print(h.shape,hs_pop.shape)
            h = torch.cat([h, hs_pop], dim=1)
            h = module(h, emb, context)
        #h = #h.type(x.dtype)
        h = self.out(h)
#         h = self.last(h)

        # Normalize output
        if self.marginal_prob_std:
            h = h / self.marginal_prob_std(t)[:, None, None, None].to(h.device)
        return h
    
class UNetTransformerArchResBlock2(nn.Module):
    def __init__(self, marginal_prob_std, in_channels=1, model_channels=32, channels=[32, 64, 128, 256], embed_dim=256, res_num=2,dropout=0,context_dim=512,
                   text_dim=512, context_embeding=512,resblock_updown=False, num_heads=1, image_size=256, psp_ratio=2, use_psp=False,dims=2,conv_resample=True):
        super().__init__()
#         self.time_embed = nn.Sequential(
#             GaussianFourierProjection(embed_dim=embed_dim),
            
#             nn.Linear(embed_dim, embed_dim)
#         )
#         self.time = get_timestep_embedding(,embed_dim)
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(embed_dim,
                            embed_dim),
            torch.nn.Linear(embed_dim,
                            embed_dim),
        ])
        self.embed_dim = embed_dim
        # n,in_channels,h,w --> n,model_channels,h,w
        self.first_block = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
        ])
        ch = model_channels
        input_block_chans = [model_channels]
        for idx, up_channel in enumerate(channels):
            for res_i in range(res_num):
                layers = [ResBlock2(in_channels=ch,
                                       out_channels=up_channel,
                                       temb_channels=embed_dim,
                                       dropout=dropout
                    )]
                ch = up_channel
                input_block_chans.append(ch)
                if idx!=0:
                # if res_i == 0 and idx%2==0:
                    layers.append(SpatialTransformer(ch, context_dim, num_heads=num_heads))
                self.first_block.append(TimestepEmbedSequential(*layers))
            
            out_ch = ch
            self.first_block.append(TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            ways="down",
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )))
            input_block_chans.append(ch)
            
        self.middle_block = TimestepEmbedSequential(
            ResBlock2(in_channels=ch,temb_channels=embed_dim, dropout=dropout,out_channels=ch),
            SpatialTransformer(ch, context_dim, num_heads=num_heads),
            ResBlock2(in_channels=ch,temb_channels=embed_dim, dropout=dropout,out_channels=ch))
            
        self.last_block = nn.ModuleList([])
        for idx, up_channel in list(enumerate(channels))[::-1]:
            for i in range(res_num+1):
                ich = input_block_chans.pop()
                layers = [ResBlock2(
                            in_channels=ch+ich,
                            temb_channels=embed_dim,
                            dropout=dropout,
                            out_channels=up_channel,
                    )]
                ch = up_channel
                if idx!=0:
                # if i == 0 and idx%2==0:
                    layers.append(
                        SpatialTransformer(ch, context_dim, num_heads=num_heads)
                    )
                out_ch = ch
                if i == 0:
                    layers.append(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                ways="up",
                            )
                            if resblock_updown
                            else Upsample(
                                ch, conv_resample, dims=dims, out_channels=out_ch
                            )

                        )
                self.last_block.append(TimestepEmbedSequential(*layers))
        self.out = nn.Sequential(
            nn.GroupNorm(32, num_channels=ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1, bias=False),
        )
        self.last = nn.Tanh()
        # The swish activation function
        self.act = nn.SiLU() # lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        if context_dim is not None:
            self.cond_embed = nn.Linear(context_embeding, context_dim)
  
    def forward(self, x, t, y=None): 
        # Obtain the Gaussian random feature embedding for t   
        #emb = self.time_embed(t)
        temb = get_timestep_embedding(t, self.embed_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        emb = self.temb.dense[1](temb)
        hs = []
        if y is not None:
#             if len(y.shape)<2:
#                 y = y.unsqueeze(-1)
            context = self.cond_embed(y)#.unsqueeze(1)
        else:
            context = y
        h = x#x.type(self.dtype)
        for module in self.first_block:
            #print("dd",module)
            h = module(h, emb, context)
            hs.append(h)
            # print(h.shape)
        h = self.middle_block(h, emb, context)
        for module in self.last_block:
            hs_pop = hs.pop()
            # print(h.shape,hs_pop.shape)
            h = torch.cat([h, hs_pop], dim=1)
            h = module(h, emb, context)
        #h = #h.type(x.dtype)
        h = self.out(h)
#         h = self.last(h)

        # Normalize output
        if self.marginal_prob_std:
            h = h / self.marginal_prob_std(t)[:, None, None, None].to(h.device)
        return h
        
        
class UNetTransformerArchResBlockEmbeding(nn.Module):
    def __init__(self, marginal_prob_std, in_channels=1, model_channels=32, channels=[32, 64, 128, 256], embed_dim=256, res_num=2,dropout=0,context_dim=512,
                   text_dim=512, nclass=512,resblock_updown=False, num_heads=1, image_size=256, psp_ratio=2, use_psp=False,dims=2,conv_resample=True):
        super().__init__()
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        # n,in_channels,h,w --> n,model_channels,h,w
        self.first_block = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
        ])
        ch = model_channels
        input_block_chans = [model_channels]
        for idx, up_channel in enumerate(channels):
            for _ in range(res_num):
                layers = [ResBlock(
                            ch,
                            embed_dim,
                            dropout,
                            out_channels=up_channel,
                            dims=dims,
                    )]
                ch = up_channel
                input_block_chans.append(ch)
                if idx!=0:
                    layers.append(SpatialTransformer(ch, context_dim, num_heads=num_heads))
                self.first_block.append(TimestepEmbedSequential(*layers))
            
            out_ch = ch
            self.first_block.append(TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            ways="down",
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )))
            input_block_chans.append(ch)
            
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch,embed_dim, dropout,dims=dims),
            SpatialTransformer(ch, context_dim, num_heads=num_heads),
            ResBlock(ch,embed_dim,dropout,dims=dims))
        
        self.last_block = nn.ModuleList([])
        for idx, up_channel in list(enumerate(channels))[::-1]:
            for i in range(res_num+1):
                ich = input_block_chans.pop()
                layers = [ResBlock(
                            ch+ich,
                            embed_dim,
                            dropout,
                            out_channels=up_channel,
                            dims=dims,
                    )]
                ch = up_channel
                if idx!=0:
                    layers.append(
                        SpatialTransformer(ch, context_dim, num_heads=num_heads)
                    )
                out_ch = ch
                if i == 0:
                    layers.append(
                            ResBlock(
                                ch,
                                embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                ways="up",
                            )
                            if resblock_updown
                            else Upsample(
                                ch, conv_resample, dims=dims, out_channels=out_ch
                            )

                        )
                self.last_block.append(TimestepEmbedSequential(*layers))
        self.out = nn.Sequential(
            nn.GroupNorm(32, num_channels=ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1, bias=False),
        )
        # The swish activation function
        self.act = nn.SiLU() # lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        if context_dim is not None:
            self.cond_embed = nn.Embedding(nclass, context_dim)
  
    def forward(self, x, t, y=None): 
        # Obtain the Gaussian random feature embedding for t   
        emb = self.time_embed(t)
        hs = []
        if y is not None:
            #print(y.shape,self.cond_embed)
            y = y.long()
            y = y.reshape(-1,1)
            context = self.cond_embed(y)
        else:
            context = y
        h = x#x.type(self.dtype)
        for module in self.first_block:
            #print("dd",module)
            h = module(h, emb, context)
            hs.append(h)
            # print(h.shape)
        h = self.middle_block(h, emb, context)
        for module in self.last_block:
            hs_pop = hs.pop()
            # print(h.shape,hs_pop.shape)
            h = torch.cat([h, hs_pop], dim=1)
            h = module(h, emb, context)
        #h = #h.type(x.dtype)
        h = self.out(h)

        # Normalize output
        if self.marginal_prob_std:
            h = h / self.marginal_prob_std(t)[:, None, None, None].to(h.device)
        return h
                        
    
    
class UNet_Tranformer_Arch_Embeding(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, input_channels=1, channels=[32, 64, 128, 256], embed_dim=256, 
                   text_dim=512, text_embeding=512, num_heads=1, image_size=256, psp_ratio=2, use_psp=False):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings of time.
          text_dim:  the embedding dimension of text / digits. 
          nClass:    number of classes you want to model.
        """
        super().__init__()
        self.use_psp = use_psp
        # Gaussian random feature embedding layer for time
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
            )
        # Encoding layers where the resolution decreases
        if use_psp:
            pool_sizes = [int(image_size/(2**i)) for i in range(1,4)]
            self.psp1 = PSPModule(channels[0], psp_ratio*channels[0], pool_sizes=pool_sizes)
            self.psp2 = PSPModule(channels[1], psp_ratio*channels[1], pool_sizes=pool_sizes)
            self.psp3 = PSPModule(channels[2], psp_ratio*channels[2], pool_sizes=pool_sizes)
            self.psp4 = PSPModule(channels[3], psp_ratio*channels[3], pool_sizes=pool_sizes)
            self.tpsp4 = PSPModule(channels[2], psp_ratio*channels[2], pool_sizes=pool_sizes)
            self.tpsp3 = PSPModule(channels[1], psp_ratio*channels[1], pool_sizes=pool_sizes)
            self.tpsp2 = PSPModule(channels[0], psp_ratio*channels[0], pool_sizes=pool_sizes)
            # self.tpsp1 = PSPModule(channels[0], 2*channels[0])
        self.conv1 = nn.Conv2d(input_channels, channels[0], 3, stride=1,padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.attn3 = SpatialTransformer(channels[2], text_dim, num_heads=num_heads) 

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    
        # YOUR CODE: interleave some attention layers with conv layers
        self.attn4 = SpatialTransformer(channels[3], text_dim, num_heads=num_heads)                        ######################################

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, padding=1,output_padding=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])   

        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, padding=1,bias=False, output_padding=1)     #  + channels[2]
#         self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, bias=False) 
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, padding=1,bias=False, output_padding=1)     #  + channels[1]
#         self.tconv2 = nn.ConvTranspose2d(channels[1], channels[0], 3, stride=2, bias=False)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.Conv2d(channels[0], input_channels, 3, stride=1,padding=1) #  + channels[0]

        # The swish activation function
        self.act = nn.SiLU() # lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        if text_dim is not None:
            self.cond_embed = nn.Linear(text_embeding, text_dim)
  
    def forward(self, x, t, y=None): 
        # Obtain the Gaussian random feature embedding for t   
        embed = self.act(self.time_embed(t)) 
        if y is not None:
            y_embed = self.cond_embed(y).unsqueeze(1)
        else:
            y_embed = y.unsqueeze(1)
            
#         print("y_embed:",y_embed.shape,y.shape)
        # Encoding path
        h1 = self.conv1(x) + self.dense1(embed) 
        ## Incorporate information from t
        ## Group normalization
        h1 = self.act(self.gnorm1(h1))
        if self.use_psp:
            h1 = self.psp1(h1)
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        if self.use_psp:
            h2 = self.psp2(h2)
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        if self.use_psp:
            h3 = self.psp3(h3)
        h3 = self.attn3(h3, y_embed) # Use your attention layers
#         print("h3",h3.shape)
        h4 = self.conv4(h3) + self.dense4(embed)
#         print("h4",h4.shape)
        h4 = self.act(self.gnorm4(h4))
        if self.use_psp:
            h4 = self.psp4(h4)
#         print("h4",h4.shape)
        # Your code: Use your additional attention layers! 
        h4 = self.attn4(h4, y_embed)       ##################### ATTENTION LAYER COULD GO HERE IF ATTN4 IS DEFINED

#         print("h4",h4.shape)
        # Decoding path
        h = self.tconv4(h4) + self.dense5(embed)
        ## Skip connection from the encoding path
        h = self.act(self.tgnorm4(h))
#         print("h",h.shape)
        if self.use_psp:
            h = self.tpsp4(h)
#         print("h",h.shape)
#         print(embed.shape)
#         print("h3",h3.shape)
        h = self.tconv3(h + h3) + self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        if self.use_psp:
            h = self.tpsp3(h)
#         print(h.shape)
#         print(h2.shape)
        h = self.tconv2(h + h2) + self.dense7(embed)
        h = self.act(self.tgnorm2(h))
#         print(h.shape)
        if self.use_psp:
            h = self.tpsp2(h)
        h = self.tconv1(h + h1)
#         print(h.shape)
#         print(h1.shape)
        

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
    
    
class UNet_Tranformer_ResBlock_Arch_Embeding(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, in_channels=1, dropout=0,conv_resample=True, model_channels=32, attention_resolutions=[2,4,8], channels_ratio=[1, 2, 3, 5],res_blocks=3, embed_dim=256, resblock_updown=False,context_dim=512, context_embeding=512, num_heads=4, image_size=256, psp_ratio=2, use_psp=False, dims=2, use_spatial_transformer=True):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings of time.
          text_dim:  the embedding dimension of text / digits. 
          nClass:    number of classes you want to model.
        """
        super().__init__()
        self.use_psp = use_psp
        # Gaussian random feature embedding layer for time
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
            )
        # Encoding layers where the resolution decreases
        if use_psp:
            pool_sizes = [int(image_size/(2**i)) for i in range(1,4)]
            self.psp1 = PSPModule(channels[0], psp_ratio*channels[0], pool_sizes=pool_sizes)
            self.psp2 = PSPModule(channels[1], psp_ratio*channels[1], pool_sizes=pool_sizes)
            self.psp3 = PSPModule(channels[2], psp_ratio*channels[2], pool_sizes=pool_sizes)
            self.psp4 = PSPModule(channels[3], psp_ratio*channels[3], pool_sizes=pool_sizes)
            self.tpsp4 = PSPModule(channels[2], psp_ratio*channels[2], pool_sizes=pool_sizes)
            self.tpsp3 = PSPModule(channels[1], psp_ratio*channels[1], pool_sizes=pool_sizes)
            self.tpsp2 = PSPModule(channels[0], psp_ratio*channels[0], pool_sizes=pool_sizes)
            # self.tpsp1 = PSPModule(channels[0], 2*channels[0])
        self.first_block = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
        ])
        ch = model_channels
        input_block_chans = [model_channels]
        ds = 1
        for level, channel_ratio in enumerate(channels_ratio):
            for _ in range(res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        embed_dim,
                        dropout,
                        out_channels=channel_ratio * model_channels,
                        dims=dims,
                    )
                ]
                ch = channel_ratio * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        SpatialTransformer(ch, context_dim, num_heads=num_heads)
                    )
                input_block_chans.append(ch)
                self.first_block.append(TimestepEmbedSequential(*layers))
            
            if level != len(channels_ratio) - 1:
                out_ch = ch
                self.first_block.append(TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            ways="down",
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                )
                    
                )
                ch = out_ch
                ds *= 2
                input_block_chans.append(ch)
        
        self.middle_block = TimestepEmbedSequential(
             ResBlock(ch,embed_dim, dropout,dims=dims),
             AttentionBlock(ch,use_checkpoint=use_checkpoint,num_heads=num_heads,num_head_channels=dim_head,use_new_attention_order=use_new_attention_order) if not use_spatial_transformer else SpatialTransformer(ch, context_dim, num_heads=num_heads),
             ResBlock(ch,embed_dim,dropout,dims=dims))
        self.output_blocks = nn.ModuleList([])
        for level, channel_ratio in list(enumerate(channels_ratio))[::-1]:
            for i in range(res_blocks+1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch+ich,
                        embed_dim,
                        dropout,
                        out_channels=channel_ratio * model_channels,
                        dims=dims,
                    )
                ]
                ch = channel_ratio * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        SpatialTransformer(ch, context_dim, num_heads=num_heads)
                    )
                if level and  i == res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            ways="up",
                        )
                        if resblock_updown
                        else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
            
        self.out = nn.Sequential(
            nn.GroupNorm(32, num_channels=ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1, bias=False),
        )
        # The swish activation function
        self.act = nn.SiLU() # lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        if context_dim is not None:
            self.cond_embed = nn.Linear(context_embeding, context_dim)
  
    def forward(self, x, t, y=None): 
        # Obtain the Gaussian random feature embedding for t   
        emb = self.time_embed(t)
        hs = []
        if y is not None:
            context = self.cond_embed(y)#.unsqueeze(1)
            if len(context.shape)<3:
                context = context.unsqueeze(0)
        else:
            context = y
        h = x#x.type(self.dtype)
        for module in self.first_block:
            #print("dd",module)
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        #h = #h.type(x.dtype)
        h = self.out(h)

        # Normalize output
        if self.marginal_prob_std:
            h = h / self.marginal_prob_std(t)[:, None, None, None].to(h.device)
        return h