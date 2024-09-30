import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .utils import window_partition, window_unpartition, add_decomposed_rel_pos, patch_partition, patch_unpartition
    
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


# code adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
# and https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py
class SpatialAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings (per channel/marker)."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = embedding_dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embedding_dim, embedding_dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class SpatialAttentionBlock(nn.Module):
    """Attention block with spatial (per-marker) attention on pixel level."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 1024,
        mlp_bias: bool = True,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(embedding_dim)
        self.attn = SpatialAttention(
            embedding_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(embedding_dim)
        self.mlp = MLP(embedding_dim=embedding_dim, mlp_dim=mlp_dim, mlp_bias=mlp_bias, act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W, E = x.shape
        shortcut = x

        x = x.reshape(B * C, H, W, E)
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = x.reshape(B, C, H, W, E)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class ChannelAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings (across channels/markers)."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = embedding_dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embedding_dim, embedding_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _ = x.shape
        # qkv with shape (3, B, nHead, C, E)
        qkv = self.qkv(x).reshape(B, C, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, C, E)
        q, k, v = qkv.reshape(3, B * self.num_heads, C, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, C, -1).permute(0, 2, 1, 3).reshape(B, C, -1)
        x = self.proj(x)

        return x


class CrossChannelAttentionBlock(nn.Module):
    """Attention block with cross-channel (per-pixel) attention on patch level."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 1024,
        mlp_bias: bool = True,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        patch_size: int = 16,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            patch_size (int): Patch size for attention blocks.
        """
        super().__init__()
        self.norm1 = norm_layer(embedding_dim)
        self.attn = ChannelAttention(
            embedding_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=False,
            rel_pos_zero_init=False,
        )

        self.norm2 = norm_layer(embedding_dim)
        self.mlp = MLP(embedding_dim=embedding_dim, mlp_dim=mlp_dim, mlp_bias=mlp_bias, act=act_layer)

        self.patch_size = patch_size
        self.patch_proj = nn.Parameter(torch.randn(patch_size * patch_size, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W, E = x.shape

        patched_x, pad_hw = patch_partition(x, self.patch_size) # [B * num_patches, C, patch_size*patch_size, E]

        x = torch.einsum("bcpe,pe->bce", patched_x, self.patch_proj) # [B * num_patches, C, E]
        x = x + self.norm1(self.attn(x))

        x = x.unsqueeze(2).expand_as(patched_x)
        x = patched_x + self.mlp(self.norm2(x))

        x = patch_unpartition(x, self.patch_size, pad_hw, (B, C, H, W, E))
        return x



class MultiplexViT(nn.Module):
    """Multiplex Vision Transformer with spatial and cross-channel attention blocks."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_dim: int = 1024,
        mlp_bias: bool = True,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        patch_size: int = 16,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            num_layers (int): Number of attention blocks.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            patch_size (int): Patch size for attention blocks.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        
