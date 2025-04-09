import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type, List

# from utils import window_partition, window_unpartition, patch_partition, patch_unpartition


class GlobalResponseNormalization(nn.Module):
    """Global Response Normalization (GRN) layer 
    from https://arxiv.org/pdf/2301.00808"""

    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, H, W, E = x.shape

        gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNextBlock(nn.Module):
    """ConvNext2 block"""
    def __init__(
            self,
            dim: int,
            inter_dim: int = None,
    ):
            super().__init__()
            inter_dim = inter_dim or dim * 4
            self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
            self.ln = nn.LayerNorm(dim)
            self.conv2 = nn.Linear(dim, inter_dim) # equivalent to nn.Conv2d(dim, inter_dim, kernel_size=1)
            self.act = nn.GELU()
            self.grn = GlobalResponseNormalization(inter_dim)
            self.conv3 = nn.Linear(inter_dim, dim) # equivalent to nn.Conv2d(inter_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, C, H, W = x.shape
        residual = x
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.ln(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.conv3(x)
        x = x.permute(0, 3, 1, 2) # [B, C, H, W]
        x = x + residual

        return x


class ConvNextBlocks(nn.Module):
    """Spatial convolutional blocks (per channel/marker)"""

    def __init__(
        self,
        embedding_dim: int,
        num_blocks: int = 1,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_blocks (int): Number of spatial convolutional blocks.
        """
        super().__init__()

        self.spatial_conv_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.spatial_conv_blocks.append(
                ConvNextBlock(
                    dim=embedding_dim,
                    inter_dim=embedding_dim*4,
                )
            )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, C, H, W  = x.shape

        # Spatial convolutional blocks
        for block in self.spatial_conv_blocks:
            x = block(x)

        return x
    

class CustomConvNeXT(nn.Module):
    """Custom ConvNeXT model with spatial convolutional blocks."""

    def __init__(
        self,
        layers_blocks: List[int],
        embedding_dims: List[int],
        channel_embedding_dim: int,
        include_stem: bool = True,
    ) -> None:
        """
        Args:
            layers_blocks (List[int]): Number of attention blocks in each layer.
            embedding_dims (List[int]): Number of input channels in each layer.
            channel_embedding_dim (int): Embedding dimension per channel pixel.
            include_stem (bool): If True, include a stem convolutional layer.
        """
        super().__init__()

        self.include_stem = include_stem
        self.poolings = nn.ModuleList()
        if include_stem:
            self.poolings.append(nn.Conv2d(channel_embedding_dim, embedding_dims[0], kernel_size=2, padding=0, stride=2))
        else:
            self.poolings.append(nn.Identity()) 

        for i, out_dim in enumerate(embedding_dims[1:]):
            input_dim = embedding_dims[i]
            self.poolings.append(
                nn.Conv2d(input_dim, out_dim, kernel_size=2, padding=0, stride=2)
            )

        # self.norms = nn.ModuleList()
        # for dim in embedding_dims:
        #     self.norms.append(nn.BatchNorm2d(dim))
        self.act = nn.GELU()
             
        self.blocks = nn.ModuleList()
        for blocks, dim in zip(layers_blocks, embedding_dims):
            self.blocks.append(
                ConvNextBlocks(
                    num_blocks=blocks,
                    embedding_dim=dim,
                )
            )
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ConvNeXT.

        Args:
            x (torch.Tensor): Multiplex images batch tensor with shape [B, C, H, W]

        Returns:
            torch.Tensor: Embedding tensor
        """

        for pooling, block in zip(self.poolings, self.blocks):
            x = self.act(pooling(x))
            x = block(x)

        return x

class MLP(nn.Module):
    """Standard MLP module"""
    def __init__(
            self, 
            embedding_dim: int,
            mlp_dim: int,
            mlp_bias: bool = True,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim, bias=mlp_bias),
            act(),
            nn.Linear(mlp_dim, embedding_dim, bias=mlp_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AttnBlock(nn.Module):
    def __init__(
            self, 
            embedding_dim : int, 
            num_heads: int, 
            mlp_ratio: float=4.,
            mlp_bias: bool=True,
        ):
        super(AttnBlock, self).__init__()
        self.num_heads = num_heads
        self.dim = embedding_dim

        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            batch_first=True
        )

        self.proj = nn.Linear(embedding_dim, embedding_dim)

        self.mlp = MLP(
            embedding_dim=embedding_dim,
            mlp_dim=int(embedding_dim * mlp_ratio),
            mlp_bias=mlp_bias
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.proj(x)
        x += res

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x += res

        return x
